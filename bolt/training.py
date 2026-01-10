"""Training pipeline for BOLT.

Components:
1. Exemplar pool parsing (individual Q/A pairs from training JSON or txt file)
2. APE instruction generation (reuses APEGenerator from lipo module)
3. VAE training with fixed K=8 exemplars
4. HbBoPs-style multi-fidelity evaluation
"""

import json
import random
import re
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from bolt.config import BOLTConfig
from bolt.encoder import GTREncoder, StructureAwareVAE

# Reuse APE generator from lipo module
from lipo.training import APEGenerator


@dataclass
class QAPair:
    """Single Q/A pair from exemplar pool."""
    question: str
    answer: str
    block_id: int  # Original exemplar block ID
    pair_id: int   # Index within block (0-4)
    pool_id: int   # Global pool index

    def format(self) -> str:
        """Format as Q/A string."""
        return f"Q: {self.question}\nA: {self.answer}"


@dataclass
class TrainingSample:
    """Training sample for VAE."""
    instruction_id: int
    instruction_text: str
    exemplar_ids: List[int]  # Pool indices
    num_exemplars: int
    error_rate: float
    fidelity: int


def parse_exemplar_pool(file_path: str) -> List[QAPair]:
    """Parse exemplar file into individual Q/A pairs.

    Args:
        file_path: Path to examples_25.txt or similar

    Returns:
        List of QAPair objects (125 pairs for 25 blocks × 5 pairs)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If no Q/A pairs found in file
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Exemplar pool file not found: {file_path}\n"
            f"Expected format: Q:/A: pairs separated by '=' dividers.\n"
            f"See datasets/hbbops/examples_25.txt for reference."
        ) from None
    except PermissionError:
        raise PermissionError(
            f"Cannot read exemplar pool file (permission denied): {file_path}"
        ) from None

    qa_pairs = []
    pool_id = 0
    block_id = 0

    # Split by block separator
    blocks = content.split("=" * 80)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Parse Q/A pairs within block (skip comment lines)
        lines = block.split("\n")
        current_q = None
        block_start_pool_id = pool_id

        for line in lines:
            line = line.strip()
            if line.startswith("#"):
                continue
            if line.startswith("Q:"):
                current_q = line[2:].strip()
            elif line.startswith("A:") and current_q:
                qa_pairs.append(QAPair(
                    question=current_q,
                    answer=line[2:].strip(),
                    block_id=block_id,
                    pair_id=pool_id - block_start_pool_id,
                    pool_id=pool_id,
                ))
                pool_id += 1
                current_q = None

        if pool_id > block_start_pool_id:
            block_id += 1

    if not qa_pairs:
        raise ValueError(
            f"No Q/A pairs found in exemplar file: {file_path}\n"
            f"Expected format: Lines starting with 'Q:' followed by 'A:'\n"
            f"Found {len(blocks)} blocks but none contained valid Q/A pairs."
        )

    return qa_pairs


def load_qa_pool_from_json(
    file_path: str,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> List[QAPair]:
    """Load Q/A pairs from training JSON file.

    Args:
        file_path: Path to train.json (list of {"question": ..., "answer": ...})
        max_samples: Maximum number of Q/A pairs to load (None = all)
        shuffle: Whether to shuffle before selecting
        seed: Random seed for reproducibility

    Returns:
        List of QAPair objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid or has wrong structure
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Q/A pool file not found: {file_path}\n"
            f"Expected format: JSON array of {{'question': ..., 'answer': ...}} objects."
        ) from None
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in Q/A pool file: {file_path}\n"
            f"Parse error at line {e.lineno}: {e.msg}"
        ) from e

    if not isinstance(data, list):
        raise ValueError(
            f"Q/A pool file must contain a JSON array, got {type(data).__name__}: {file_path}"
        )

    if not data:
        raise ValueError(f"Q/A pool file is empty: {file_path}")

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(data)

    if max_samples is not None:
        data = data[:max_samples]

    qa_pairs = []
    for pool_id, item in enumerate(data):
        qa_pairs.append(QAPair(
            question=item["question"],
            answer=item["answer"],
            block_id=pool_id,  # Each example is its own "block"
            pair_id=0,
            pool_id=pool_id,
        ))

    return qa_pairs


def generate_ape_instructions(
    model: str,
    backend: str,
    validation_data: List[dict],
    num_instructions: int = 2000,
    cache_path: Optional[str] = None,
    force_regenerate: bool = False,
    verbose: bool = True,
) -> List[str]:
    """Generate instructions using APE (Automatic Prompt Engineer).

    Args:
        model: LLM model for generation
        backend: LLM backend (vllm, openai, etc.)
        validation_data: Task examples for prompt building
        num_instructions: Target number of instructions
        cache_path: Path to cache file (None = no caching)
        force_regenerate: Force regeneration even if cache exists
        verbose: Print progress

    Returns:
        List of generated instructions
    """
    generator = APEGenerator(model=model, backend=backend)

    if cache_path:
        return generator.generate_or_load(
            cache_path=cache_path,
            validation_data=validation_data,
            num_instructions=num_instructions,
            force_regenerate=force_regenerate,
            verbose=verbose,
            augment=True,  # Enable augmentation for diversity
        )
    else:
        return generator.generate(
            validation_data=validation_data,
            num_instructions=num_instructions,
            verbose=verbose,
            augment=True,
        )


def load_instructions(file_path: str) -> List[str]:
    """Load instructions from file.

    Handles numbered format like "1. Answer:" or plain lines.

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If no instructions found
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Instructions file not found: {file_path}\n"
            f"Expected format: One instruction per line, optionally numbered."
        ) from None

    instructions = []
    for line in content.split("\n"):
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue
        # Remove numbering like "1. " or "25. "
        match = re.match(r"^\d+\.\s*(.+)$", line)
        if match:
            instructions.append(match.group(1))
        else:
            instructions.append(line)

    if not instructions:
        raise ValueError(f"No instructions found in file: {file_path}")

    return instructions


# Number pattern for answer extraction
NUMBER_PATTERN = r'[-+]?\d+(?:[.,]\d+)?'


def extract_answer(text: str) -> Optional[str]:
    """Extract last number from model output."""
    if not text:
        return None
    numbers = re.findall(NUMBER_PATTERN, text)
    return numbers[-1] if numbers else None


def compare_numbers(predicted: str, ground_truth: str, tolerance: float = 1e-6) -> bool:
    """Compare two numbers with tolerance."""
    if predicted == ground_truth:
        return True
    try:
        pred_num = float(predicted.replace(',', ''))
        gt_num = float(ground_truth.replace(',', ''))
        return abs(pred_num - gt_num) <= tolerance
    except (ValueError, TypeError):
        return False


class HbBoPsEvaluator:
    """Multi-fidelity evaluator for prompt combinations."""

    def __init__(
        self,
        model: str,
        backend: str,
        validation_path: str,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.model = model
        self.backend = backend
        self.device = device

        # Load validation data
        with open(validation_path, "r") as f:
            self.validation_data = json.load(f)

        # Shuffle once for consistent fidelity subsets (reproducible with seed)
        rng = random.Random(seed)
        rng.shuffle(self.validation_data)

        self._llm_client = None

    @property
    def llm_client(self):
        """Lazy load LLM client."""
        if self._llm_client is None:
            from src.llm_client import create_llm_client
            self._llm_client = create_llm_client(self.model, backend=self.backend)
        return self._llm_client

    def evaluate(
        self,
        instruction: str,
        qa_pairs: List[QAPair],
        fidelity: int,
    ) -> float:
        """Evaluate prompt at given fidelity.

        Args:
            instruction: Instruction text
            qa_pairs: Selected Q/A pairs
            fidelity: Number of validation samples

        Returns:
            error_rate: Fraction incorrect
        """
        # Build exemplar string
        exemplar_text = "\n\n".join(qa.format() for qa in qa_pairs) if qa_pairs else ""

        # Get subset of validation data
        subset = self.validation_data[:fidelity]

        # Build prompts for each validation example
        prompts = []
        for ex in subset:
            if exemplar_text:
                prompt = f"{exemplar_text}\n\nQuestion: {ex['question']}\n\n{instruction}\n\nAnswer:"
            else:
                prompt = f"Question: {ex['question']}\n\n{instruction}\n\nAnswer:"
            prompts.append(prompt)

        # Generate responses
        try:
            responses = self.llm_client.generate_batch(prompts, max_tokens=1024)
        except (ConnectionError, TimeoutError) as e:
            # Network errors - re-raise to let caller decide retry strategy
            raise RuntimeError(f"Network error during LLM evaluation: {e}") from e
        except Exception as e:
            print(f"LLM error during evaluation:\n{traceback.format_exc()}")
            raise RuntimeError(f"LLM evaluation failed: {e}") from e

        # Validate response count matches input
        if len(responses) != len(subset):
            raise ValueError(
                f"Length mismatch: got {len(responses)} responses but {len(subset)} samples. "
                f"This may indicate LLM API issues or token truncation."
            )

        # Count errors
        errors = 0
        for i, ex in enumerate(subset):
            # Extract gold answer
            gold = re.findall(NUMBER_PATTERN, ex['answer'])
            gold = gold[-1] if gold else None

            # Extract prediction
            pred = extract_answer(responses[i])

            if gold is None or pred is None or not compare_numbers(pred, gold):
                errors += 1

        return errors / len(subset) if subset else 1.0


def compute_selection_targets(
    design_data: List[Dict],
    n_pool: int,
) -> Dict[int, torch.Tensor]:
    """Create binary mask targets from Hyperband results.

    For each instruction, find the best-performing exemplar set
    and create a binary mask indicating which exemplars to select.

    Args:
        design_data: List of dicts from Hyperband with instruction_id,
                     exemplar_ids, error_rate
        n_pool: Total number of exemplars in pool

    Returns:
        Dict mapping instruction_id → binary mask (n_pool,)
    """
    from collections import defaultdict

    # Group by instruction_id
    by_instruction = defaultdict(list)
    for entry in design_data:
        by_instruction[entry['instruction_id']].append(entry)

    targets = {}
    for inst_id, entries in by_instruction.items():
        # Find best exemplar set for this instruction (lowest error)
        best_entry = min(entries, key=lambda e: e['error_rate'])

        # Create binary mask
        mask = torch.zeros(n_pool)
        for ex_id in best_entry['exemplar_ids']:
            mask[ex_id] = 1.0
        targets[inst_id] = mask

    return targets


class VAETrainer:
    """Trainer for Structure-Aware VAE (Simplified with fixed K=8)."""

    def __init__(
        self,
        vae: StructureAwareVAE,
        gtr_encoder: GTREncoder,
        qa_pool: List[QAPair],
        instructions: List[str],
        config: BOLTConfig,
        selection_targets: Optional[Dict[int, torch.Tensor]] = None,
    ):
        self.vae = vae
        self.gtr_encoder = gtr_encoder
        self.qa_pool = qa_pool
        self.instructions = instructions
        self.config = config
        self.selection_targets = selection_targets or {}

        # Training stats (populated after train())
        self.training_stats: dict = {}

        # Pre-compute embeddings
        print("Pre-computing pool embeddings...")
        pool_texts = [qa.format() for qa in qa_pool]
        self.pool_embeddings = self.gtr_encoder.encode(pool_texts).clone().detach()
        self.n_pool = len(qa_pool)

        print("Pre-computing instruction embeddings...")
        self.instruction_embeddings = self.gtr_encoder.encode(instructions).clone().detach()

    def prepare_batch(
        self,
        samples: List[TrainingSample],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch tensors from samples.

        Returns:
            instruction_embs: (batch, 768)
            exemplar_embs: (batch, K, 768) - K=8 fixed
            exemplar_mask: (batch, K) - all True for K=8
            target_exemplar_mask: (batch, N_pool) - binary mask of good exemplars
        """
        batch_size = len(samples)
        K = self.config.num_exemplars  # Fixed K=8
        device = self.pool_embeddings.device

        # Instruction embeddings
        inst_ids = [s.instruction_id for s in samples]
        instruction_embs = self.instruction_embeddings[inst_ids]

        # Exemplar embeddings (fixed K=8)
        exemplar_embs = torch.zeros(batch_size, K, 768, device=device)
        exemplar_mask = torch.ones(batch_size, K, dtype=torch.bool, device=device)

        # Target exemplar masks for selection loss
        target_exemplar_mask = torch.zeros(batch_size, self.n_pool, device=device)

        for b, sample in enumerate(samples):
            # Fill exemplar embeddings (pad with zeros if < K)
            for s, pool_idx in enumerate(sample.exemplar_ids[:K]):
                exemplar_embs[b, s] = self.pool_embeddings[pool_idx]

            # If sample has fewer than K exemplars, mask out padding
            if sample.num_exemplars < K:
                exemplar_mask[b, sample.num_exemplars:] = False

            # Target mask from selection_targets (if available) or from sample
            if sample.instruction_id in self.selection_targets:
                target_exemplar_mask[b] = self.selection_targets[sample.instruction_id].to(device)
            else:
                # Fallback: use this sample's exemplars as target
                for pool_idx in sample.exemplar_ids:
                    target_exemplar_mask[b, pool_idx] = 1.0

        return instruction_embs, exemplar_embs, exemplar_mask, target_exemplar_mask

    def train(
        self,
        samples: List[TrainingSample],
        epoch_callback: Optional[Callable[[int, Dict[str, float]], bool]] = None,
    ) -> Dict[str, List[float]]:
        """Train VAE on samples (no curriculum, fixed K=8).

        Args:
            samples: Training samples
            epoch_callback: Optional callback called after each epoch.
                Receives (epoch, metrics_dict) and returns True if training should stop.
                Used for ASHA pruning integration.

        Returns:
            history: Training metrics
        """
        device = self.pool_embeddings.device
        self.vae = self.vae.to(device)
        self.vae.train()

        optimizer = torch.optim.AdamW(
            self.vae.parameters(),
            lr=self.config.vae_lr,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.vae_epochs,
            eta_min=self.config.vae_eta_min,
        )

        history = {
            "total": [],
            "recon": [],
            "kl": [],
            "selection": [],
            "cosine_mean": [],
        }

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.vae_epochs):
            # KL annealing
            if epoch < self.config.vae_annealing_epochs:
                current_beta = self.config.vae_beta * (epoch / self.config.vae_annealing_epochs)
            else:
                current_beta = self.config.vae_beta

            # Shuffle samples (no curriculum filtering - always K=8)
            shuffled_samples = samples.copy()
            random.shuffle(shuffled_samples)

            epoch_losses = {k: 0.0 for k in history}
            num_batches = 0

            # Mini-batches
            for batch_start in range(0, len(shuffled_samples), self.config.vae_batch_size):
                batch = shuffled_samples[batch_start:batch_start + self.config.vae_batch_size]
                if len(batch) == 0:
                    continue

                (
                    instruction_embs,
                    exemplar_embs,
                    exemplar_mask,
                    target_exemplar_mask,
                ) = self.prepare_batch(batch)

                optimizer.zero_grad()
                loss, loss_dict = self.vae(
                    instruction_emb=instruction_embs,
                    exemplar_embs=exemplar_embs,
                    exemplar_mask=exemplar_mask,
                    pool_embeddings=self.pool_embeddings,
                    target_exemplar_mask=target_exemplar_mask,
                    beta=current_beta,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.vae.parameters(),
                    self.config.vae_grad_clip,
                )
                optimizer.step()

                epoch_losses["total"] += loss_dict["total"]
                epoch_losses["recon"] += loss_dict["recon"]
                epoch_losses["kl"] += loss_dict["kl_total"]
                epoch_losses["selection"] += loss_dict["selection"]
                epoch_losses["cosine_mean"] += loss_dict["cosine_mean"]
                num_batches += 1

            scheduler.step()

            # Average losses
            for k in epoch_losses:
                epoch_losses[k] /= max(num_batches, 1)
                history[k].append(epoch_losses[k])

            # Early stopping (after annealing)
            if epoch >= self.config.vae_annealing_epochs:
                if epoch_losses["total"] < best_loss:
                    best_loss = epoch_losses["total"]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.vae_patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            # Logging
            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}: "
                    f"loss={epoch_losses['total']:.4f}, "
                    f"recon={epoch_losses['recon']:.4f}, "
                    f"kl={epoch_losses['kl']:.4f}, "
                    f"sel={epoch_losses['selection']:.4f}, "
                    f"cos={epoch_losses['cosine_mean']:.4f}, "
                    f"beta={current_beta:.4f}"
                )

            # ASHA pruning callback
            if epoch_callback is not None:
                should_stop = epoch_callback(epoch + 1, epoch_losses)
                if should_stop:
                    print(f"PRUNED at epoch {epoch + 1} by ASHA")
                    break

        # Collect training stats
        self._collect_training_stats(
            epochs_trained=epoch + 1,
            final_loss=best_loss,
            early_stopped=patience_counter >= self.config.vae_patience,
            num_samples=len(samples),
            history=history,
        )

        return history

    def _collect_training_stats(
        self,
        epochs_trained: int,
        final_loss: float,
        early_stopped: bool,
        num_samples: int,
        history: Dict[str, List[float]],
    ):
        """Collect training statistics for debugging and analysis."""
        # Sample history to reduce size (keep ~100 points per metric)
        sample_rate = max(1, len(history["total"]) // 100)

        self.training_stats = {
            "epochs_trained": epochs_trained,
            "final_loss": float(final_loss),
            "early_stopped": early_stopped,
            "num_samples": num_samples,
            "config": {
                "vae_beta": self.config.vae_beta,
                "vae_lr": self.config.vae_lr,
                "instruction_latent_dim": self.config.instruction_latent_dim,
                "exemplar_latent_dim": self.config.exemplar_latent_dim,
                "num_exemplars": self.config.num_exemplars,
            },
            # Sampled loss curves
            "loss_history": {
                "total": history["total"][::sample_rate],
                "recon": history["recon"][::sample_rate],
                "kl": history["kl"][::sample_rate],
                "selection": history["selection"][::sample_rate],
                "cosine_mean": history["cosine_mean"][::sample_rate],
            },
            # Final values
            "final_recon_loss": history["recon"][-1] if history["recon"] else None,
            "final_kl_loss": history["kl"][-1] if history["kl"] else None,
            "final_selection_loss": history["selection"][-1] if history["selection"] else None,
            "final_cosine_mean": history["cosine_mean"][-1] if history["cosine_mean"] else None,
        }
