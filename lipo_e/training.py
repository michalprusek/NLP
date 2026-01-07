"""Training pipeline for LIPO-E.

Components:
1. Exemplar pool parsing (individual Q/A pairs from training JSON or txt file)
2. APE instruction generation (reuses lipo.training.APEGenerator)
3. Training data generation with variable K selection
4. VAE training with curriculum learning
5. HbBoPs-style multi-fidelity evaluation
"""

import os
import json
import random
import re
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from lipo_e.config import LIPOEConfig
from lipo_e.encoder import GTREncoder, StructureAwareVAE

# Reuse APE generator from lipo
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
        pair_id = 0
        found_qa = False

        for line in lines:
            line = line.strip()
            # Skip comments
            if line.startswith("#"):
                continue
            if line.startswith("Q:"):
                current_q = line[2:].strip()
            elif line.startswith("A:") and current_q:
                answer = line[2:].strip()
                qa_pairs.append(QAPair(
                    question=current_q,
                    answer=answer,
                    block_id=block_id,
                    pair_id=pair_id,
                    pool_id=pool_id,
                ))
                pool_id += 1
                pair_id += 1
                current_q = None
                found_qa = True

        if found_qa:
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


def build_prompt(instruction: str, qa_pairs: List[QAPair]) -> str:
    """Build evaluation prompt from instruction and Q/A pairs.

    Format (Q_end style from OPRO):
        [exemplars]

        Q: {test_question}
        {instruction}
        A:
    """
    parts = []

    # Add exemplars
    for qa in qa_pairs:
        parts.append(qa.format())

    if parts:
        parts.append("")  # Blank line before test question

    # Instruction comes after question in Q_end format
    # (actual question will be added during evaluation)
    prompt = "\n\n".join(parts)

    return prompt, instruction


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
            # Log the full error and re-raise - don't return fake data
            import traceback
            print(f"LLM error during evaluation:\n{traceback.format_exc()}")
            raise RuntimeError(f"LLM evaluation failed: {e}") from e

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


def generate_training_data(
    instructions: List[str],
    qa_pool: List[QAPair],
    evaluator: HbBoPsEvaluator,
    config: LIPOEConfig,
    output_path: str,
) -> List[TrainingSample]:
    """Generate training data with variable exemplar selection.

    Samples combinations of (instruction, {qa_pairs}, K) and evaluates them.

    Args:
        instructions: List of instruction texts
        qa_pool: List of Q/A pairs
        evaluator: Multi-fidelity evaluator
        config: Configuration
        output_path: Path to save results

    Returns:
        List of TrainingSample
    """
    samples = []
    cache_key = set()  # Avoid duplicate evaluations

    num_instructions = len(instructions)
    num_pool = len(qa_pool)

    # Generate diverse samples
    for _ in tqdm(range(config.num_training_samples), desc="Generating training data"):
        # Random instruction
        inst_id = random.randint(0, num_instructions - 1)

        # Random number of exemplars
        K = random.randint(config.min_exemplars, config.num_slots)

        # Random exemplar selection (no replacement)
        if K > 0:
            exemplar_ids = sorted(random.sample(range(num_pool), K))
        else:
            exemplar_ids = []

        # Check cache
        cache_tuple = (inst_id, tuple(exemplar_ids))
        if cache_tuple in cache_key:
            continue
        cache_key.add(cache_tuple)

        # Get Q/A pairs
        qa_pairs = [qa_pool[i] for i in exemplar_ids]

        # Evaluate (use high fidelity for training data)
        fidelity = len(evaluator.validation_data)  # Full fidelity
        error_rate = evaluator.evaluate(
            instruction=instructions[inst_id],
            qa_pairs=qa_pairs,
            fidelity=fidelity,
        )

        sample = TrainingSample(
            instruction_id=inst_id,
            instruction_text=instructions[inst_id],
            exemplar_ids=exemplar_ids,
            num_exemplars=K,
            error_rate=error_rate,
            fidelity=fidelity,
        )
        samples.append(sample)

        # Save incrementally
        if len(samples) % 50 == 0:
            save_samples(samples, output_path)

    # Final save
    save_samples(samples, output_path)

    return samples


def save_samples(samples: List[TrainingSample], path: str):
    """Save training samples to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(s) for s in samples], f, indent=2)


def load_samples(path: str) -> List[TrainingSample]:
    """Load training samples from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    return [TrainingSample(**d) for d in data]


class VAETrainer:
    """Trainer for Structure-Aware VAE."""

    def __init__(
        self,
        vae: StructureAwareVAE,
        gtr_encoder: GTREncoder,
        qa_pool: List[QAPair],
        instructions: List[str],
        config: LIPOEConfig,
    ):
        self.vae = vae
        self.gtr_encoder = gtr_encoder
        self.qa_pool = qa_pool
        self.instructions = instructions
        self.config = config

        # Pre-compute embeddings (clone to allow gradients during training)
        print("Pre-computing pool embeddings...")
        pool_texts = [qa.format() for qa in qa_pool]
        self.pool_embeddings = self.gtr_encoder.encode(pool_texts).clone().detach()

        print("Pre-computing instruction embeddings...")
        self.instruction_embeddings = self.gtr_encoder.encode(instructions).clone().detach()

    def prepare_batch(
        self,
        samples: List[TrainingSample],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch tensors from samples.

        Returns:
            instruction_embs: (batch, 768)
            exemplar_embs: (batch, max_K, 768)
            exemplar_mask: (batch, max_K)
            true_indices: (batch, max_K)
            true_counts: (batch,)
        """
        batch_size = len(samples)
        max_K = max(s.num_exemplars for s in samples)
        if max_K == 0:
            max_K = 1  # At least one slot

        device = self.pool_embeddings.device

        # Instruction embeddings
        inst_ids = [s.instruction_id for s in samples]
        instruction_embs = self.instruction_embeddings[inst_ids]

        # Exemplar embeddings and masks
        exemplar_embs = torch.zeros(batch_size, max_K, 768, device=device)
        exemplar_mask = torch.zeros(batch_size, max_K, dtype=torch.bool, device=device)
        true_indices = torch.zeros(batch_size, max_K, dtype=torch.long, device=device)
        true_counts = torch.zeros(batch_size, dtype=torch.long, device=device)

        for b, sample in enumerate(samples):
            K = sample.num_exemplars
            true_counts[b] = K
            for s, pool_idx in enumerate(sample.exemplar_ids):
                exemplar_embs[b, s] = self.pool_embeddings[pool_idx]
                exemplar_mask[b, s] = True
                true_indices[b, s] = pool_idx

        return instruction_embs, exemplar_embs, exemplar_mask, true_indices, true_counts

    def train(
        self,
        samples: List[TrainingSample],
    ) -> Dict[str, List[float]]:
        """Train VAE on samples.

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
            "num_ex": [],
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

            # Curriculum: gradually increase max exemplars
            if self.config.vae_curriculum and epoch < self.config.vae_curriculum_epochs:
                progress = epoch / self.config.vae_curriculum_epochs
                max_ex_curriculum = int(1 + progress * (self.config.num_slots - 1))
                filtered_samples = [s for s in samples if s.num_exemplars <= max_ex_curriculum]
            else:
                filtered_samples = samples

            # Shuffle
            random.shuffle(filtered_samples)

            epoch_losses = {k: 0.0 for k in history}
            num_batches = 0

            # Mini-batches
            for batch_start in range(0, len(filtered_samples), self.config.vae_batch_size):
                batch = filtered_samples[batch_start:batch_start + self.config.vae_batch_size]
                if len(batch) == 0:
                    continue

                (
                    instruction_embs,
                    exemplar_embs,
                    exemplar_mask,
                    true_indices,
                    true_counts,
                ) = self.prepare_batch(batch)

                optimizer.zero_grad()
                loss, loss_dict = self.vae(
                    instruction_emb=instruction_embs,
                    exemplar_embs=exemplar_embs,
                    exemplar_mask=exemplar_mask,
                    pool_embeddings=self.pool_embeddings,
                    true_exemplar_indices=true_indices,
                    true_num_exemplars=true_counts,
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
                epoch_losses["num_ex"] += loss_dict["num_exemplars"]
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

        return history
