"""InvBO inference pipeline for LIPO.

Provides:
- Vec2TextInverter: Embedding-to-text inverter (32_tokens by default, 512_tokens optional)
- LIPOHyperbandInference: Complete inference with UCB/LogEI optimization

Self-contained within lipo package. Uses external libs: vec2text, safetensors, huggingface_hub.
"""

import re
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable


def is_valid_instruction(text: str, min_length: int = 10) -> bool:
    """Validate instruction text quality.

    Rejects instructions with:
    - Unicode garbage characters (common Vec2Text artifacts)
    - Too short length (< min_length characters)
    - Excessive whitespace (> 50% whitespace)
    - Repetitive patterns (same word 4+ times in a row)

    Args:
        text: Instruction text to validate
        min_length: Minimum acceptable length (default: 10 chars)

    Returns:
        True if instruction passes all quality checks
    """
    if not text:
        return False

    # Reject unicode garbage (common Vec2Text artifacts from 512_tokens model)
    # These characters indicate encoding issues in the inversion
    unicode_garbage = r'[»«â€¢™®©†‡°±²³µ¶·¹º¼½¾¿×÷]'
    if re.search(unicode_garbage, text):
        return False

    # Reject too short
    stripped = text.strip()
    if len(stripped) < min_length:
        return False

    # Reject mostly whitespace (> 50% whitespace)
    non_whitespace = len(stripped.replace(' ', '').replace('\t', '').replace('\n', ''))
    if non_whitespace < len(stripped) * 0.5:
        return False

    # Reject repetitive patterns (same word 4+ times consecutively)
    words = stripped.lower().split()
    if len(words) >= 4:
        for i in range(len(words) - 3):
            if words[i] == words[i+1] == words[i+2] == words[i+3]:
                return False

    return True

from lipo.config import Config, get_device
from lipo.encoder import GTRInstructionEncoder, InstructionVAE
from lipo.gp import GPWithEI
from lipo.instruction import InstructionOnlyPrompt
from lipo.turbo import (
    TrustRegionManager,
    PotentialAwareAnchorSelector,
    create_turbo_manager,
    create_pas_selector,
)
from lipo.quality_kpi import compute_gp_spearman, compute_system_gap, format_kpi_report


@dataclass
class InversionResult:
    """Result of latent-to-text inversion."""
    instruction_text: str
    latent: torch.Tensor
    embedding: torch.Tensor
    cosine_similarity: float
    predicted_error: float
    ei_value: float


@dataclass
class IterationRecord:
    """Record of a single optimization iteration."""
    iteration: int
    instruction: str
    cosine_similarity: float
    predicted_error: float
    actual_error: Optional[float]
    improved: bool
    best_error_so_far: float
    gp_samples: int
    log_ei: Optional[float] = None
    rejection_attempts: int = 0  # How many candidates were rejected before acceptance
    low_quality_accepted: bool = False  # Whether this was forced acceptance below threshold
    # Optimization Gap Test metrics (z_opt vs z_real after Vec2Text inversion)
    z_opt_z_real_cosine: float = 0.0     # Cosine sim in VAE latent space (32D)
    z_opt_z_real_euclidean: float = 0.0  # Euclidean distance in VAE latent space
    z_opt_z_real_gp_cosine: float = 0.0  # Same as z_opt_z_real_cosine (no adapter, GP on 32D)
    predicted_error_at_z_real: float = 0.0  # GP prediction at actual z_real point
    # TuRBO trust region state
    trust_region_length: float = 0.0  # Current trust region side length
    trust_region_action: str = ""  # Action taken: "none", "expand", "shrink", "restart"
    anchor_idx: int = -1  # Index of selected anchor in training data
    # Adaptive UCB and noise injection
    ucb_beta: float = 0.0  # UCB beta used for this iteration
    noise_applied: bool = False  # Whether latent noise was applied


class Vec2TextInverter:
    """Vec2Text embedding-to-text inverter.

    Model type is configurable via model_type parameter.
    Default from config: "32_tokens" (recommended, no unicode issues).
    Alternative: "512_tokens" (longer sequences but produces garbage characters).

    Supports fine-tuned models via finetuned_path parameter.
    """

    def __init__(
        self,
        num_steps: int = 50,
        beam_width: int = 8,
        max_length: int = 128,
        device: str = "auto",
        model_type: str = "32_tokens",
        finetuned_path: Optional[str] = None,
        finetuned_inverter_path: Optional[str] = None,
    ):
        """Initialize inverter.

        Args:
            num_steps: Max new tokens for generation
            beam_width: Beam search width
            max_length: Maximum output length
            device: Device to use
            model_type: "32_tokens" (recommended) or "512_tokens"
            finetuned_path: Path to fine-tuned corrector model (None = use pre-trained)
            finetuned_inverter_path: Path to fine-tuned InversionModel (None = use pre-trained)
        """
        if model_type not in ("32_tokens", "512_tokens"):
            raise ValueError(f"model_type must be '32_tokens' or '512_tokens'")

        self.num_steps = num_steps
        self.beam_width = beam_width
        self.max_length = max_length
        self.device = get_device(device)
        self.model_type = model_type
        self.finetuned_path = finetuned_path
        self.finetuned_inverter_path = finetuned_inverter_path
        self._corrector = None
        self._inversion_model = None
        self._finetuned_model = None  # Fine-tuned T5 model
        self._finetuned_tokenizer = None
        self._finetuned_inverter = None  # Fine-tuned InversionModel
        self._pre_reload_callback = None  # Called before reload to free GPU memory

    def _load_model(self):
        """Lazy load Vec2Text model.

        Priority:
        1. Fine-tuned InversionModel (best, trained on instructions)
        2. Fine-tuned Corrector (original hypothesis + T5 correction)
        3. Pre-trained 32_tokens or 512_tokens model
        """
        # Check if fine-tuned InversionModel should be loaded (highest priority)
        if self.finetuned_inverter_path:
            self._load_finetuned_inverter()
        # Check if fine-tuned corrector model should be loaded
        elif self.finetuned_path:
            self._load_finetuned()
        elif self.model_type == "32_tokens":
            self._load_32_tokens()
        else:
            self._load_512_tokens()

    def _load_finetuned(self):
        """Load fine-tuned T5 corrector model."""
        if self._finetuned_model is not None:
            return

        from pathlib import Path
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        finetuned_path = Path(self.finetuned_path)
        if not finetuned_path.exists():
            raise ValueError(f"Fine-tuned model path does not exist: {finetuned_path}")

        print(f"Loading fine-tuned Vec2Text corrector from {finetuned_path}...")

        self._finetuned_model = T5ForConditionalGeneration.from_pretrained(
            str(finetuned_path)
        )
        self._finetuned_tokenizer = T5Tokenizer.from_pretrained(str(finetuned_path))

        self._finetuned_model = self._finetuned_model.to(self.device).eval()
        print(f"  Fine-tuned Vec2Text loaded on {self.device}")

        # Also load the original inversion model for generating initial hypotheses
        self._load_32_tokens_inversion_only()

    def _load_finetuned_inverter(self):
        """Load fine-tuned InversionModel.

        The fine-tuned InversionModel directly maps embeddings to text,
        without needing a corrector stage. This provides best results
        when fine-tuned on instruction data.
        """
        if self._finetuned_inverter is not None:
            return

        from pathlib import Path
        import torch
        from vec2text.models.config import InversionConfig
        from vec2text.models.inversion import InversionModel

        finetuned_path = Path(self.finetuned_inverter_path)
        if not finetuned_path.exists():
            raise ValueError(f"Fine-tuned InversionModel path does not exist: {finetuned_path}")

        print(f"Loading fine-tuned Vec2Text InversionModel from {finetuned_path}...")

        # Load config from pre-trained (fine-tuning doesn't change architecture)
        inv_config = InversionConfig.from_pretrained(
            "ielabgroup/vec2text_gtr-base-st_inversion"
        )

        # Create model and load fine-tuned weights
        self._finetuned_inverter = InversionModel(inv_config)

        # Load weights from pytorch_model.bin
        weights_path = finetuned_path / "pytorch_model.bin"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            load_result = self._finetuned_inverter.load_state_dict(state_dict, strict=False)
            if load_result.missing_keys or load_result.unexpected_keys:
                print(f"WARNING: Fine-tuned model weight mismatch at {finetuned_path}")
                if load_result.missing_keys:
                    print(f"  Missing keys: {load_result.missing_keys[:5]}{'...' if len(load_result.missing_keys) > 5 else ''}")
                if load_result.unexpected_keys:
                    print(f"  Unexpected keys: {load_result.unexpected_keys[:5]}{'...' if len(load_result.unexpected_keys) > 5 else ''}")
        else:
            # Try safetensors format
            from safetensors.torch import load_file
            safetensors_path = finetuned_path / "model.safetensors"
            if safetensors_path.exists():
                state_dict = load_file(str(safetensors_path))
                load_result = self._finetuned_inverter.load_state_dict(state_dict, strict=False)
                if load_result.missing_keys or load_result.unexpected_keys:
                    print(f"WARNING: Fine-tuned model weight mismatch at {finetuned_path}")
                    if load_result.missing_keys:
                        print(f"  Missing keys: {load_result.missing_keys[:5]}{'...' if len(load_result.missing_keys) > 5 else ''}")
                    if load_result.unexpected_keys:
                        print(f"  Unexpected keys: {load_result.unexpected_keys[:5]}{'...' if len(load_result.unexpected_keys) > 5 else ''}")
            else:
                raise ValueError(
                    f"No weights found at {finetuned_path}. "
                    f"Expected pytorch_model.bin or model.safetensors"
                )

        self._finetuned_inverter = self._finetuned_inverter.to(self.device).eval()
        print(f"  Fine-tuned InversionModel loaded on {self.device}")

    def _load_32_tokens(self):
        """Load ielabgroup Vec2Text with corrector (32 token limit)."""
        if self._corrector is not None:
            return

        import vec2text
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        from vec2text.models.config import InversionConfig
        from vec2text.models.inversion import InversionModel
        from vec2text.models.corrector_encoder import CorrectorEncoderModel

        print("Loading Vec2Text (32_tokens with corrector)...")

        inv_weights = hf_hub_download(
            "ielabgroup/vec2text_gtr-base-st_inversion", "model.safetensors"
        )
        inv_config = InversionConfig.from_pretrained(
            "ielabgroup/vec2text_gtr-base-st_inversion"
        )
        inversion_model = InversionModel(inv_config)
        inv_load_result = inversion_model.load_state_dict(load_file(inv_weights), strict=False)
        if inv_load_result.missing_keys or inv_load_result.unexpected_keys:
            print(f"WARNING: InversionModel weight mismatch")
            if inv_load_result.missing_keys:
                print(f"  Missing keys: {inv_load_result.missing_keys[:5]}{'...' if len(inv_load_result.missing_keys) > 5 else ''}")
            if inv_load_result.unexpected_keys:
                print(f"  Unexpected keys: {inv_load_result.unexpected_keys[:5]}{'...' if len(inv_load_result.unexpected_keys) > 5 else ''}")
        inversion_model = inversion_model.to(self.device).eval()

        corr_weights = hf_hub_download(
            "ielabgroup/vec2text_gtr-base-st_corrector", "model.safetensors"
        )
        corr_config = InversionConfig.from_pretrained(
            "ielabgroup/vec2text_gtr-base-st_corrector"
        )
        corrector_model = CorrectorEncoderModel(corr_config)
        corr_load_result = corrector_model.load_state_dict(load_file(corr_weights), strict=False)
        if corr_load_result.missing_keys or corr_load_result.unexpected_keys:
            print(f"WARNING: CorrectorModel weight mismatch")
            if corr_load_result.missing_keys:
                print(f"  Missing keys: {corr_load_result.missing_keys[:5]}{'...' if len(corr_load_result.missing_keys) > 5 else ''}")
            if corr_load_result.unexpected_keys:
                print(f"  Unexpected keys: {corr_load_result.unexpected_keys[:5]}{'...' if len(corr_load_result.unexpected_keys) > 5 else ''}")
        corrector_model = corrector_model.to(self.device).eval()

        self._corrector = vec2text.load_corrector(inversion_model, corrector_model)
        print(f"  Vec2Text (32_tokens) loaded on {self.device}")

    def _load_32_tokens_inversion_only(self):
        """Load only the inversion model (for generating hypotheses for fine-tuned corrector)."""
        if self._inversion_model is not None:
            return

        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
        from vec2text.models.config import InversionConfig
        from vec2text.models.inversion import InversionModel

        print("Loading InversionModel for hypothesis generation...")

        inv_weights = hf_hub_download(
            "ielabgroup/vec2text_gtr-base-st_inversion", "model.safetensors"
        )
        inv_config = InversionConfig.from_pretrained(
            "ielabgroup/vec2text_gtr-base-st_inversion"
        )
        self._inversion_model = InversionModel(inv_config)
        load_result = self._inversion_model.load_state_dict(load_file(inv_weights), strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(f"WARNING: InversionModel weight mismatch")
            if load_result.missing_keys:
                print(f"  Missing keys: {load_result.missing_keys[:5]}{'...' if len(load_result.missing_keys) > 5 else ''}")
            if load_result.unexpected_keys:
                print(f"  Unexpected keys: {load_result.unexpected_keys[:5]}{'...' if len(load_result.unexpected_keys) > 5 else ''}")
        self._inversion_model = self._inversion_model.to(self.device).eval()
        print(f"  InversionModel loaded on {self.device}")

    def _load_512_tokens(self):
        """Load Vec2Text InversionModel (512 token limit)."""
        if self._inversion_model is not None:
            return

        import os
        import json
        from safetensors.torch import load_file
        from huggingface_hub import snapshot_download
        from vec2text.models.config import InversionConfig
        from vec2text.models.inversion import InversionModel

        print("Loading Vec2Text (512_tokens InversionModel)...")

        # Clear CUDA cache before loading - Vec2Text loads its own GTR model internally
        # Note: gc.collect() must come before empty_cache() to free Python objects first
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        model_dir = snapshot_download("vec2text/gtr-512-noise-0.00001")
        config = InversionConfig.from_pretrained(model_dir)
        print(f"  Config: max_seq_length={config.max_seq_length}")

        self._inversion_model = InversionModel(config)

        # Load sharded weights
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)

        shard_files = set(index["weight_map"].values())
        state_dict = {}
        for shard_file in shard_files:
            shard_path = os.path.join(model_dir, shard_file)
            shard_dict = load_file(shard_path)
            state_dict.update(shard_dict)

        load_result = self._inversion_model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(f"WARNING: Vec2Text 512_tokens model weight mismatch")
            if load_result.missing_keys:
                print(f"  Missing keys: {load_result.missing_keys[:5]}{'...' if len(load_result.missing_keys) > 5 else ''}")
            if load_result.unexpected_keys:
                print(f"  Unexpected keys: {load_result.unexpected_keys[:5]}{'...' if len(load_result.unexpected_keys) > 5 else ''}")
        self._inversion_model = self._inversion_model.to(self.device).eval()

        print(f"  Vec2Text (512_tokens) loaded on {self.device}")

    def _ensure_on_device(self, pre_reload_callback: callable = None):
        """Ensure model is on the correct device.

        Reloads from CPU if model was offloaded. This allows Vec2Text to be
        offloaded during evaluation and automatically reloaded for next inversion.

        Args:
            pre_reload_callback: Optional callback to call BEFORE reloading to GPU.
                                 Used to free GPU memory (e.g., shutdown vLLM evaluator).
        """
        needs_reload = False
        # self.device can be str or torch.device - normalize to device type string
        target_type = self.device.type if hasattr(self.device, 'type') else str(self.device).split(':')[0]

        if self._inversion_model is not None:
            # Check if 512_tokens model needs to be moved to GPU
            try:
                param = next(self._inversion_model.parameters())
                if param.device.type == 'cpu' and target_type == 'cuda':
                    needs_reload = True
            except StopIteration:
                print("WARNING: _inversion_model has no parameters - this may indicate a loading problem")

        if self._corrector is not None:
            # Check if 32_tokens model needs to be moved to GPU
            if hasattr(self._corrector, 'model') and self._corrector.model is not None:
                try:
                    param = next(self._corrector.model.parameters())
                    if param.device.type == 'cpu' and target_type == 'cuda':
                        needs_reload = True
                except StopIteration:
                    print("WARNING: _corrector.model has no parameters - this may indicate a loading problem")

        if self._finetuned_model is not None:
            # Check if fine-tuned model needs to be moved to GPU
            try:
                param = next(self._finetuned_model.parameters())
                if param.device.type == 'cpu' and target_type == 'cuda':
                    needs_reload = True
            except StopIteration:
                print("WARNING: _finetuned_model has no parameters - this may indicate a loading problem")

        if self._finetuned_inverter is not None:
            # Check if fine-tuned Inverter needs to be moved to GPU
            try:
                param = next(self._finetuned_inverter.parameters())
                if param.device.type == 'cpu' and target_type == 'cuda':
                    needs_reload = True
            except StopIteration:
                print("WARNING: _finetuned_inverter has no parameters - this may indicate a loading problem")

        if needs_reload:
            if pre_reload_callback is not None:
                pre_reload_callback()
            self.reload()

    def set_pre_reload_callback(self, callback: callable):
        """Set callback to be called before reloading model to GPU.

        This is used to free GPU memory (e.g., shutdown vLLM evaluator)
        before reloading Vec2Text to GPU.
        """
        self._pre_reload_callback = callback

    def invert(self, embedding: torch.Tensor) -> str:
        """Invert embedding to text.

        Args:
            embedding: 768D GTR embedding

        Returns:
            Reconstructed text
        """
        self._load_model()

        # Auto-reload to GPU if model was offloaded to CPU
        # Use pre-reload callback to free GPU memory (e.g., shutdown vLLM evaluator)
        self._ensure_on_device(self._pre_reload_callback)

        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        embedding = embedding.to(self.device)

        # Use fine-tuned models if available (priority order)
        if self.finetuned_inverter_path and self._finetuned_inverter is not None:
            return self._invert_finetuned_inverter(embedding)
        elif self.finetuned_path and self._finetuned_model is not None:
            return self._invert_finetuned(embedding)
        elif self.model_type == "32_tokens":
            return self._invert_32_tokens(embedding)
        else:
            return self._invert_512_tokens(embedding)

    def _invert_32_tokens(self, embedding: torch.Tensor) -> str:
        """Invert using ielabgroup corrector."""
        import vec2text

        result = vec2text.invert_embeddings(
            embeddings=embedding,
            corrector=self._corrector,
            num_steps=self.num_steps,
            sequence_beam_width=self.beam_width,
        )
        return result[0] if isinstance(result, list) else result

    def _invert_512_tokens(self, embedding: torch.Tensor) -> str:
        """Invert using InversionModel direct generation."""
        gen_kwargs = {
            "num_beams": self.beam_width,
            "max_length": self.max_length,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.2,
        }

        with torch.no_grad():
            output_ids = self._inversion_model.generate(
                inputs={"frozen_embeddings": embedding},
                generation_kwargs=gen_kwargs,
            )

        tokenizer = self._inversion_model.tokenizer
        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return result.strip()

    def _invert_finetuned_inverter(self, embedding: torch.Tensor) -> str:
        """Invert using fine-tuned InversionModel.

        Direct inversion without corrector stage - the fine-tuned model
        learned to directly produce high-quality text from embeddings.

        Args:
            embedding: 768D GTR embedding (batch of 1)

        Returns:
            Instruction text
        """
        gen_kwargs = {
            "num_beams": self.beam_width,
            "max_length": self.max_length,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.2,
        }

        with torch.no_grad():
            output_ids = self._finetuned_inverter.generate(
                inputs={"frozen_embeddings": embedding},
                generation_kwargs=gen_kwargs,
            )

        tokenizer = self._finetuned_inverter.tokenizer
        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return result.strip()

    def _invert_finetuned(self, embedding: torch.Tensor) -> str:
        """Invert using fine-tuned T5 corrector model.

        Pipeline:
        1. InversionModel generates initial hypothesis from embedding
        2. Fine-tuned T5 model corrects the hypothesis

        Args:
            embedding: 768D GTR embedding (batch of 1)

        Returns:
            Corrected instruction text
        """
        # Step 1: Generate initial hypothesis using InversionModel
        with torch.no_grad():
            gen_kwargs = {
                "num_beams": 4,
                "max_length": 64,
                "no_repeat_ngram_size": 3,
            }
            output_ids = self._inversion_model.generate(
                inputs={"frozen_embeddings": embedding},
                generation_kwargs=gen_kwargs,
            )

        tokenizer = self._inversion_model.tokenizer
        hypothesis = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Step 2: Correct hypothesis using fine-tuned T5 model
        input_text = f"Correct: {hypothesis}"
        inputs = self._finetuned_tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._finetuned_model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.beam_width,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        result = self._finetuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip()

    def invert_with_adaptive_correction(
        self,
        embedding: torch.Tensor,
        gtr_encoder: "GTRInstructionEncoder",
        max_corrections: int = 100,
        cosine_threshold: float = 0.98,
        verbose: bool = False,
    ) -> Tuple[str, dict]:
        """Invert embedding to text with adaptive iterative correction.

        Uses early stopping based on cosine similarity to target embedding.
        Instead of running a fixed number of corrector steps, stops when
        the reconstructed text's embedding is close enough to the target.

        This is task-agnostic and VAE-agnostic - it only looks at how well
        the current text matches the target embedding in GTR space.

        Pipeline:
            1. Inverter generates initial hypothesis (uses fine-tuned if available)
            2. Loop until cosine >= threshold or max_corrections reached:
               a. Encode current text with GTR
               b. Check cosine similarity to target embedding
               c. If good enough, stop
               d. Otherwise, run one corrector step (pre-trained corrector)

        Args:
            embedding: Target 768D GTR embedding to invert
            gtr_encoder: GTR encoder for re-encoding intermediate results
            max_corrections: Maximum correction iterations (default: 100)
            cosine_threshold: Stop when cosine >= this value (default: 0.98)
            verbose: Print progress during correction

        Returns:
            Tuple of (reconstructed_text, stats_dict) where stats_dict contains:
            - iterations: Number of correction steps used
            - final_cosine: Final cosine similarity achieved
            - early_stopped: Whether stopped before max_corrections
            - cosine_history: List of cosine similarities at each step
        """
        self._load_model()
        self._ensure_on_device(self._pre_reload_callback)

        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        embedding = embedding.to(self.device)

        # Need the 32_tokens corrector model for iterative correction
        if self._corrector is None:
            self._load_32_tokens()

        corrector = self._corrector
        tokenizer = corrector.tokenizer

        # Step 1: Generate initial hypothesis
        # Use fine-tuned inverter if available (trained on noisy data = more robust)
        # Otherwise fall back to pre-trained inverter from corrector
        if self._finetuned_inverter is not None:
            if verbose:
                print("  Using fine-tuned InversionModel for initial hypothesis")
            with torch.no_grad():
                gen_kwargs = {
                    "num_beams": self.beam_width,
                    "max_length": self.max_length,
                    "no_repeat_ngram_size": 3,
                }
                hypothesis_input_ids = self._finetuned_inverter.generate(
                    inputs={"frozen_embeddings": embedding},
                    generation_kwargs=gen_kwargs,
                )
            # Need to embed using corrector's embedder for compatibility
            hypothesis_embedding = corrector.embed_generated_hypothesis(
                input_ids=hypothesis_input_ids
            )
            hypothesis_attention_mask = (
                hypothesis_input_ids != corrector.model.encoder_decoder.config.pad_token_id
            ).int()
            frozen_embeddings = embedding
        else:
            if verbose:
                print("  Using pre-trained InversionModel for initial hypothesis")
            # Use corrector's internal method for pre-trained inverter
            with torch.no_grad():
                (
                    frozen_embeddings,
                    hypothesis_input_ids,
                    hypothesis_attention_mask,
                    hypothesis_embedding,
                ) = corrector._get_hypothesis_uncached(inputs={"frozen_embeddings": embedding})

        current_text = tokenizer.decode(hypothesis_input_ids[0], skip_special_tokens=True).strip()

        # Check initial quality using external GTR (for task-agnostic evaluation)
        current_emb = gtr_encoder.encode_tensor(current_text).to(self.device)
        initial_cosine = F.cosine_similarity(
            embedding, current_emb.unsqueeze(0)
        ).item()

        cosine_history = [initial_cosine]

        if verbose:
            print(f"  Initial inversion cosine: {initial_cosine:.4f}")

        # Early stop if already good enough
        if initial_cosine >= cosine_threshold:
            return current_text, {
                "iterations": 0,
                "final_cosine": initial_cosine,
                "early_stopped": True,
                "cosine_history": cosine_history,
            }

        # Step 2: Iterative correction with early stopping
        corrector_model = corrector.model

        gen_kwargs = {
            "num_beams": self.beam_width,
            "max_length": self.max_length,
            "no_repeat_ngram_size": 3,
            "early_stopping": False,
            "do_sample": False,
        }

        # Track best result (corrector can oscillate/degrade)
        best_text = current_text
        best_cosine = initial_cosine
        best_step = 0
        plateau_count = 0
        plateau_threshold = 0.001  # Consider plateau if change < this
        max_plateau_steps = 20  # Stop if no improvement for this many steps

        for step in range(max_corrections):
            # Run one corrector step
            # The corrector needs: target embedding + hypothesis tokens + hypothesis embedding
            with torch.no_grad():
                corrector_output = corrector_model.generate(
                    inputs={
                        "frozen_embeddings": frozen_embeddings,
                        "hypothesis_input_ids": hypothesis_input_ids,
                        "hypothesis_attention_mask": hypothesis_attention_mask,
                        "hypothesis_embedding": hypothesis_embedding,
                    },
                    generation_kwargs=gen_kwargs,
                )

            # Update hypothesis for next iteration
            hypothesis_input_ids = corrector_output
            hypothesis_attention_mask = (
                hypothesis_input_ids != corrector_model.encoder_decoder.config.pad_token_id
            ).int()

            # Re-embed hypothesis using corrector's internal embedder
            hypothesis_embedding = corrector.embed_generated_hypothesis(
                input_ids=hypothesis_input_ids
            )

            new_text = tokenizer.decode(corrector_output[0], skip_special_tokens=True).strip()

            # Re-encode with external GTR and check similarity to target
            new_emb = gtr_encoder.encode_tensor(new_text).to(self.device)
            new_cosine = F.cosine_similarity(
                embedding, new_emb.unsqueeze(0)
            ).item()

            cosine_history.append(new_cosine)

            # Track best result
            if new_cosine > best_cosine:
                best_cosine = new_cosine
                best_text = new_text
                best_step = step + 1
                plateau_count = 0  # Reset plateau counter on improvement
            else:
                plateau_count += 1

            if verbose and (step + 1) % 10 == 0:
                print(f"  Step {step + 1}: cosine = {new_cosine:.4f}, best = {best_cosine:.4f} @ step {best_step}")

            # Check for early stopping - threshold reached
            if new_cosine >= cosine_threshold:
                if verbose:
                    print(f"  Early stop at step {step + 1}: cosine {new_cosine:.4f} >= {cosine_threshold}")
                return new_text, {
                    "iterations": step + 1,
                    "final_cosine": new_cosine,
                    "early_stopped": True,
                    "cosine_history": cosine_history,
                    "best_cosine": best_cosine,
                    "best_step": best_step,
                }

            # Check for plateau - no improvement for many steps
            if plateau_count >= max_plateau_steps:
                if verbose:
                    print(f"  Early stop at step {step + 1}: no improvement for {max_plateau_steps} steps")
                    print(f"  Returning best result from step {best_step} (cosine = {best_cosine:.4f})")
                return best_text, {
                    "iterations": step + 1,
                    "final_cosine": new_cosine,
                    "early_stopped": True,
                    "cosine_history": cosine_history,
                    "best_cosine": best_cosine,
                    "best_step": best_step,
                    "plateau_stopped": True,
                }

        # Reached max corrections - return best result (not final!)
        if verbose:
            print(f"  Max corrections reached. Returning best from step {best_step} (cosine = {best_cosine:.4f})")

        return best_text, {
            "iterations": max_corrections,
            "final_cosine": cosine_history[-1],
            "early_stopped": False,
            "cosine_history": cosine_history,
            "best_cosine": best_cosine,
            "best_step": best_step,
        }

    def offload(self):
        """Move Vec2Text model to CPU and clear CUDA cache.

        Call this before loading large models (e.g., Qwen evaluator) to free GPU memory.
        The model will be automatically reloaded to GPU on next invert() call.
        """
        import gc

        if self._corrector is not None:
            # 32_tokens model - move all components to CPU
            if hasattr(self._corrector, 'inversion_trainer') and self._corrector.inversion_trainer is not None:
                self._corrector.inversion_trainer.model = self._corrector.inversion_trainer.model.to('cpu')
            if hasattr(self._corrector, 'model') and self._corrector.model is not None:
                self._corrector.model = self._corrector.model.to('cpu')
            if hasattr(self._corrector, 'embedder') and self._corrector.embedder is not None:
                self._corrector.embedder = self._corrector.embedder.to('cpu')
            print("  Vec2Text (32_tokens) offloaded to CPU")

        if self._inversion_model is not None:
            # 512_tokens model or inversion-only - move to CPU
            self._inversion_model = self._inversion_model.to('cpu')
            print("  Vec2Text (inversion model) offloaded to CPU")

        if self._finetuned_model is not None:
            # Fine-tuned T5 model - move to CPU
            self._finetuned_model = self._finetuned_model.to('cpu')
            print("  Vec2Text (fine-tuned corrector) offloaded to CPU")

        if self._finetuned_inverter is not None:
            # Fine-tuned InversionModel - move to CPU
            self._finetuned_inverter = self._finetuned_inverter.to('cpu')
            print("  Vec2Text (fine-tuned inverter) offloaded to CPU")

        # Clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reload(self):
        """Reload Vec2Text model to GPU after offload."""
        if self._corrector is not None:
            if hasattr(self._corrector, 'inversion_trainer') and self._corrector.inversion_trainer is not None:
                self._corrector.inversion_trainer.model = self._corrector.inversion_trainer.model.to(self.device)
            if hasattr(self._corrector, 'model') and self._corrector.model is not None:
                self._corrector.model = self._corrector.model.to(self.device)
            if hasattr(self._corrector, 'embedder') and self._corrector.embedder is not None:
                self._corrector.embedder = self._corrector.embedder.to(self.device)
            print(f"  Vec2Text (32_tokens) reloaded to {self.device}")

        if self._inversion_model is not None:
            self._inversion_model = self._inversion_model.to(self.device)
            print(f"  Vec2Text (inversion model) reloaded to {self.device}")

        if self._finetuned_model is not None:
            self._finetuned_model = self._finetuned_model.to(self.device)
            print(f"  Vec2Text (fine-tuned corrector) reloaded to {self.device}")

        if self._finetuned_inverter is not None:
            self._finetuned_inverter = self._finetuned_inverter.to(self.device)
            print(f"  Vec2Text (fine-tuned inverter) reloaded to {self.device}")


def validate_roundtrip_quality(
    vae: InstructionVAE,
    gtr: GTRInstructionEncoder,
    inverter: Vec2TextInverter,
    instructions: List[str],
    n_samples: int = 20,
    verbose: bool = True,
) -> dict:
    """Validate round-trip quality of VAE + Vec2Text pipeline.

    Tests how well the full pipeline can reconstruct known instructions:
        instruction → GTR → VAE(encode→decode) → Vec2Text → GTR → cosine_sim

    Poor round-trip quality indicates GP may optimize in "empty space" -
    regions that don't correspond to valid instructions.

    Args:
        vae: Trained InstructionVAE
        gtr: GTR encoder
        inverter: Vec2Text inverter
        instructions: List of instructions to test
        n_samples: Number of random samples to test
        verbose: Print progress and results

    Returns:
        Dict with metrics:
        - mean_sim: Mean cosine similarity
        - std_sim: Std of similarities
        - min_sim: Worst reconstruction
        - poor_count: Number with sim < 0.90
        - samples: List of (original, reconstructed, sim) tuples
    """
    import random
    import numpy as np

    # Sample instructions
    n = min(n_samples, len(instructions))
    samples = random.sample(instructions, n)

    vae.eval()
    vae_dev = next(vae.parameters()).device
    sims = []
    sample_results = []

    for i, instruction in enumerate(samples):
        # Full pipeline - ensure device consistency
        emb_original = gtr.encode_tensor(instruction)
        with torch.no_grad():
            emb_for_vae = emb_original.to(vae_dev)
            mu, _ = vae.encode(emb_for_vae)
            decoded = vae.decode(mu)

        reconstructed = inverter.invert(decoded)
        emb_recon = gtr.encode_tensor(reconstructed)

        sim = F.cosine_similarity(
            emb_original.unsqueeze(0),
            emb_recon.unsqueeze(0)
        ).item()
        sims.append(sim)
        sample_results.append((instruction, reconstructed, sim))

        if verbose and (i + 1) % 5 == 0:
            print(f"  Validated {i + 1}/{n} samples...")

    sims_arr = np.array(sims)
    results = {
        "mean_sim": float(np.mean(sims_arr)),
        "std_sim": float(np.std(sims_arr)),
        "min_sim": float(np.min(sims_arr)),
        "max_sim": float(np.max(sims_arr)),
        "poor_count": int(np.sum(sims_arr < 0.90)),
        "acceptable_count": int(np.sum(sims_arr >= 0.90)),
        "n_samples": n,
        "samples": sample_results,
    }

    if verbose:
        quality = "GOOD" if results["mean_sim"] >= 0.90 else "POOR"
        print(f"""
============================================================
ROUND-TRIP DIAGNOSTIC
============================================================
Tested: {n} instructions
Mean similarity: {results['mean_sim']:.4f}
Min: {results['min_sim']:.4f}, Max: {results['max_sim']:.4f}
Below 0.90 (poor): {results['poor_count']}
Below 0.95 (acceptable): {n - results['acceptable_count'] + results['poor_count']}
Interpretation: {quality} - {"good reconstruction" if quality == "GOOD" else "significant meaning loss in reconstruction"}
  {"" if quality == "GOOD" else "This may explain lack of optimization improvement."}
""")
        # Show worst example (full text per CLAUDE.md - never truncate prompts)
        worst_idx = np.argmin(sims_arr)
        orig, recon, wsim = sample_results[worst_idx]
        print(f"Worst reconstruction (sim={wsim:.4f}):")
        print(f"  Original:\n    {orig}")
        print(f"  Reconstructed:\n    {recon}")

    return results



class LIPOHyperbandInference:
    """InvBO inference pipeline for LIPO.

    Pipeline:
        1. Optimize in 32D VAE latent space using LogEI acquisition
           (GP operates directly on 32D latent with ARD kernel)
        2. Decode optimal latent to 768D embedding via VAE decoder
        3. Invert embedding to text via Vec2Text (512_tokens)
        4. Evaluate and add to GP

    Uses 512_tokens Vec2Text model for longer instruction generation.
    Includes KPI tracking for GP quality and optimization gap monitoring.
    """

    def __init__(
        self,
        gp: GPWithEI,
        vae: InstructionVAE,
        config: Config,
        gtr: Optional[GTRInstructionEncoder] = None,
        evaluator: Optional[Callable[[str, List[dict]], float]] = None,
        validation_data: Optional[List[dict]] = None,
        initial_best_instruction: Optional[str] = None,
        initial_best_error: Optional[float] = None,
    ):
        """Initialize inference pipeline.

        Args:
            gp: Trained GPWithEI
            vae: Trained InstructionVAE
            config: Unified pipeline configuration
            gtr: GTR encoder (for validation)
            evaluator: Function (instruction, data) -> error_rate
            validation_data: Validation Q/A pairs for evaluation
            initial_best_instruction: Best instruction from Hyperband (to avoid null)
            initial_best_error: Best error from Hyperband
        """
        self.gp = gp
        self.vae = vae
        self.config = config
        # VAEWithAdapter for decoding (32D -> 768D)
        self.vae_with_adapter = gp.vae_with_adapter
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.evaluator = evaluator
        self.validation_data = validation_data

        self.gtr = gtr if gtr is not None else GTRInstructionEncoder(device=str(self.device))
        self.inverter = Vec2TextInverter(
            beam_width=config.vec2text_beam,
            device=str(self.device),
            model_type=config.vec2text_model,
            max_length=config.vec2text_max_length,
            finetuned_path=config.vec2text_finetuned_path,
            finetuned_inverter_path=config.vec2text_finetuned_inverter_path,
        )

        # History
        self.iteration_history: List[IterationRecord] = []
        self.total_llm_calls: int = 0
        self.best_error: float = float("inf")
        self.best_instruction: Optional[str] = None
        self._consecutive_retrain_failures: int = 0  # Track consecutive GP retrain failures
        self._total_retrain_failures: int = 0  # Track total GP retrain failures (never resets)

        # KPI tracking lists for periodic quality reporting
        self._predicted_errors: List[float] = []
        self._actual_errors: List[float] = []
        self._z_gaps: List[float] = []

        # Initialize best from Hyperband results or GP
        if initial_best_instruction is not None and initial_best_error is not None:
            self.best_instruction = initial_best_instruction
            self.best_error = initial_best_error
        elif gp.best_error_rate is not None:
            self.best_error = gp.best_error_rate  # Use property that returns positive error rate
            # Note: instruction not available from GP alone

        # Initialize TuRBO trust region manager
        self.use_turbo = config.turbo_enabled
        self.use_pas = config.pas_enabled

        if self.use_turbo:
            self.trust_region = create_turbo_manager(config, self.device)
        else:
            self.trust_region = None

        if self.use_pas:
            self.anchor_selector = create_pas_selector(config, self.device)
        else:
            self.anchor_selector = None

        # Distance penalty settings (used when TuRBO disabled)
        self.distance_penalty_enabled = config.distance_penalty_enabled
        self.distance_weight = config.distance_weight
        self.distance_threshold = config.distance_threshold

        # Cache global bounds (computed on first iteration)
        self._global_bounds: Optional[torch.Tensor] = None

    def _get_global_bounds(self) -> torch.Tensor:
        """Get global latent bounds from training data.

        Computes bounds on first call and caches them.

        Returns:
            Global bounds tensor, shape (2, latent_dim)
        """
        if self._global_bounds is None:
            from lipo.botorch_acq import get_latent_bounds
            self._global_bounds = get_latent_bounds(
                encoder=self.gp.vae_with_adapter,
                X_train=self.gp.X_train,
                X_min=self.gp.X_min,
                X_max=self.gp.X_max,
                margin=self.config.latent_margin,
            )
        return self._global_bounds

    def optimize_latent_botorch(
        self,
        num_restarts: int = 64,
        raw_samples: int = 4096,
        verbose: bool = True,
        seed: Optional[int] = None,
        bounds: Optional[torch.Tensor] = None,
        acquisition_type: Optional[str] = None,
        ucb_beta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Optimize VAE latent using BoTorch acquisition function.

        Supports both UCB (exploration-focused) and LogEI (exploitation-focused).

        Uses multi-start L-BFGS-B with proper gradient flow:
            z (latent_dim VAE latent) -> GP posterior -> Acquisition

        GP operates directly on VAE latent (no adapter compression).
        ARD lengthscales allow the kernel to learn which dimensions matter.

        Args:
            num_restarts: Number of L-BFGS-B restarts (default: 64)
            raw_samples: Raw samples for initialization seeding (default: 4096)
            verbose: Print progress
            seed: Optional random seed for reproducibility
            bounds: Optional custom bounds (e.g., trust region bounds). If None, uses global bounds.
            acquisition_type: "ucb" or "logei" (default: from config)
            ucb_beta: UCB exploration parameter (default: from config)

        Returns:
            (optimal_latent, acq_value) tuple where:
            - optimal_latent: Best VAE latent tensor, shape (latent_dim,)
            - acq_value: Acquisition function value at optimal point
        """
        from lipo.botorch_acq import (
            LatentSpaceAcquisition,
            get_latent_bounds,
        )

        # Use config defaults if not provided
        if acquisition_type is None:
            acquisition_type = self.config.acquisition_type
        if ucb_beta is None:
            ucb_beta = self.config.ucb_beta

        # Determine if distance penalty should be active
        # UCB already explores via σ term, so disable distance penalty for UCB
        use_distance_penalty = (
            self.distance_penalty_enabled
            and not self.use_turbo
            and acquisition_type != "ucb"
        )

        if verbose:
            if acquisition_type == "ucb":
                acq_name = f"UCB (β={ucb_beta})"
            else:
                acq_name = "LogEI"
            if use_distance_penalty:
                acq_name = f"DistancePenalized{acq_name}"
            print(f"Optimizing with BoTorch {acq_name} ({num_restarts} restarts, {raw_samples} raw samples)...")
            if use_distance_penalty:
                print(f"  Distance penalty: weight={self.distance_weight}, threshold={self.distance_threshold}")

        # Get latent bounds (use provided bounds or compute global bounds)
        if bounds is None:
            bounds = get_latent_bounds(
                encoder=self.gp.vae_with_adapter,
                X_train=self.gp.X_train,
                X_min=self.gp.X_min,
                X_max=self.gp.X_max,
                margin=self.config.latent_margin,
            )

        # Compute normalized training data for distance penalty
        X_train_normalized = None
        if use_distance_penalty:
            denom = self.gp.X_max - self.gp.X_min
            denom[denom == 0] = 1.0
            X_train_normalized = (self.gp.X_train - self.gp.X_min) / denom

        # Create acquisition optimizer
        # Optimization path: z (latent_dim) → GP → kernel (ARD) → Acquisition (+ distance penalty)
        acq_optimizer = LatentSpaceAcquisition(
            gp_model=self.gp.gp_model,
            bounds=bounds,
            device=self.device,
            X_train_normalized=X_train_normalized,
            distance_penalty_enabled=use_distance_penalty,
            distance_weight=self.distance_weight,
            distance_threshold=self.distance_threshold,
            acquisition_type=acquisition_type,
            ucb_beta=ucb_beta,
        )

        # best_f is the best observed GP target value (-min_error).
        # Since GP predicts -error_rate, qLogEI maximizes (finds lower error).
        best_f = self.gp.y_best
        z_opt, acq_value = acq_optimizer.optimize(
            best_f=best_f,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            seed=seed,
        )

        if verbose:
            acq_label = "UCB" if acquisition_type == "ucb" else "LogEI"
            print(f"  BoTorch {acq_label}: {acq_value.item():.4f}")

        # Return as 1D tensor
        return z_opt.squeeze(), acq_value.item()

    def _log_iteration_summary(
        self,
        record: IterationRecord,
        pred_std: float = 0.0,
    ) -> None:
        """Print consolidated iteration summary with all metrics.

        Centralizes all iteration metrics in one formatted output block
        for easy monitoring and debugging.

        Args:
            record: IterationRecord with all iteration data
            pred_std: Prediction standard deviation from GP
        """
        # IMPORTANT: Never truncate prompts in log output (per CLAUDE.md)
        instr_display = record.instruction

        actual_str = f"{record.actual_error:.4f}" if record.actual_error is not None else "N/A (skipped)"
        improved_str = "YES" if record.improved else "no"
        log_ei_str = f"{record.log_ei:.4f}" if record.log_ei is not None else "N/A"

        print(f"""
═══════════════════════════════════════════════════════════════
ITERATION {record.iteration} SUMMARY
═══════════════════════════════════════════════════════════════
Instruction: {instr_display}
─────────────────────────────────────────────────────────────────
PERFORMANCE METRICS:
  Predicted Error:    {record.predicted_error:.4f} ± {pred_std:.4f}
  Actual Error:       {actual_str}
  Best Error So Far:  {record.best_error_so_far:.4f}
  Improved:           {improved_str}
─────────────────────────────────────────────────────────────────
OPTIMIZATION GAP METRICS:
  VAE Latent Cosine:  {record.z_opt_z_real_cosine:.4f}
  VAE Latent L2:      {record.z_opt_z_real_euclidean:.4f}
  GP Space Cosine:    {record.z_opt_z_real_gp_cosine:.4f}
  Pred @ z_real:      {record.predicted_error_at_z_real:.4f}
─────────────────────────────────────────────────────────────────
GENERATION QUALITY:
  Cosine Similarity:  {record.cosine_similarity:.4f}
  LogEI:              {log_ei_str}
  Rejection Attempts: {record.rejection_attempts}
  Low Quality Accept: {record.low_quality_accepted}
─────────────────────────────────────────────────────────────────
GP Status:
  Training Samples:   {record.gp_samples}
═══════════════════════════════════════════════════════════════""")

    def compute_inference_kpis(self) -> dict:
        """Compute GP Spearman and System Gap KPIs from tracked data.

        Returns:
            Dictionary with:
            - gp_quality: GP prediction quality metrics
            - system_gap: Optimization gap metrics
        """
        gp_kpi = compute_gp_spearman(self._predicted_errors, self._actual_errors)
        gap_kpi = compute_system_gap(self._z_gaps)
        return {"gp_quality": gp_kpi, "system_gap": gap_kpi}

    def _log_kpi_report(self, iteration: int) -> None:
        """Log periodic KPI report.

        Called every 10 iterations to monitor optimization quality.
        """
        kpis = self.compute_inference_kpis()
        print(f"\n{format_kpi_report(kpis, iteration)}\n")

    def run_iteration(
        self,
        iteration: int,
        num_restarts: int = 64,
        raw_samples: int = 1024,
        skip_eval: bool = False,
        verbose: bool = True,
        total_iterations: int = 50,
        ucb_beta_override: Optional[float] = None,
        noise_scale: float = 0.0,
    ) -> IterationRecord:
        """Run a single optimization iteration using BoTorch qLogEI.

        Uses BoTorch's gradient-based LogEI optimization with multi-start
        L-BFGS-B for finding optimal latent points.

        Candidate Rejection:
            Enforces cosine similarity threshold between decoder(z) and GTR(text)
            to reject misaligned candidates that Vec2Text failed to reconstruct properly.
            - Candidates with cosine_sim < config.cosine_sim_threshold (default 0.90) are rejected
            - Up to config.max_rejection_attempts (default 5) attempts with different seeds
            - If all attempts fail, accepts the best candidate with a WARNING

        Args:
            iteration: Iteration number
            num_restarts: Number of L-BFGS-B restarts for BoTorch optimization
            raw_samples: Raw samples for initialization seeding
            skip_eval: Skip LLM evaluation (use GP prediction)
            verbose: Print progress
            total_iterations: Total iterations for adaptive UCB beta
            ucb_beta_override: Override UCB beta (for adaptive scheduling)
            noise_scale: Scale for latent noise injection (0 = disabled)

        Returns:
            IterationRecord with results including rejection_attempts and low_quality_accepted
        """
        if verbose:
            print(f"\n--- Iteration {iteration} ---")

        # Get thresholds from config
        cosine_sim_threshold = getattr(self.config, 'cosine_sim_threshold', 0.90)
        max_rejection_attempts = getattr(self.config, 'max_rejection_attempts', 5)

        # Initialize rejection tracking (in case loop doesn't execute)
        rejection_attempts = 0
        low_quality_accepted = False

        # === TuRBO + PAS: Select anchor and compute trust region bounds ===
        anchor_idx = -1
        turbo_action = ""
        trust_region_length = 0.0

        global_bounds = self._get_global_bounds()

        if self.use_turbo and self.trust_region is not None:
            trust_region_length = self.trust_region.state.length

            # Select anchor using PAS (Potential-Aware Selection)
            if self.use_pas and self.anchor_selector is not None:
                try:
                    anchor, anchor_idx = self.anchor_selector.select_anchor(
                        gp_model=self.gp.gp_model,
                        X_train=self.gp.X_train,
                        y_train=self.gp.y_train,
                        X_min=self.gp.X_min,
                        X_max=self.gp.X_max,
                        trust_length=trust_region_length,
                        global_bounds=global_bounds,
                        verbose=verbose,
                    )
                    self.trust_region.set_anchor(anchor)
                except RuntimeError as e:
                    # PAS failed (e.g., GP numerical issues) - fall back to best-y
                    print(f"WARNING: PAS anchor selection failed: {e}")
                    print("  Falling back to best-y anchor selection")
                    best_idx = self.gp.y_train.argmax().item()
                    denom = self.gp.X_max - self.gp.X_min
                    denom[denom == 0] = 1.0
                    anchor = (self.gp.X_train[best_idx] - self.gp.X_min) / denom
                    self.trust_region.set_anchor(anchor)
                    anchor_idx = best_idx
            else:
                # Without PAS, use best observed point as anchor
                best_idx = self.gp.y_train.argmax().item()
                denom = self.gp.X_max - self.gp.X_min
                denom[denom == 0] = 1.0
                anchor = (self.gp.X_train[best_idx] - self.gp.X_min) / denom
                self.trust_region.set_anchor(anchor)
                anchor_idx = best_idx

            # Get ARD lengthscales from GP kernel for LOL-BO style scaling
            # Lengthscales are in 32D VAE space (no adapter) - each dimension gets its own lengthscale
            lengthscales = None
            try:
                if hasattr(self.gp.gp_model, 'covar_module'):
                    base_kernel = self.gp.gp_model.covar_module.base_kernel
                    if hasattr(base_kernel, 'lengthscale'):
                        lengthscales = base_kernel.lengthscale.detach().squeeze()
            except AttributeError as e:
                # Only catch AttributeError - other exceptions (CUDA, memory) should propagate
                import warnings
                warnings.warn(
                    f"Could not extract ARD lengthscales from GP kernel: {e}. "
                    f"Falling back to uniform trust region scaling."
                )

            # Get ARD-aware trust region bounds (LOL-BO style)
            bounds = self.trust_region.get_ard_bounds(global_bounds, lengthscales)

            if verbose:
                print(f"  TuRBO: {self.trust_region.get_state_summary()}")
                if lengthscales is not None:
                    ls_str = ", ".join([f"{ls:.3f}" for ls in lengthscales[:5].tolist()])
                    print(f"  ARD lengthscales (first 5): [{ls_str}, ...]")
                print(f"  Anchor: idx={anchor_idx}, using {'PAS' if self.use_pas else 'best-y'} selection")
        else:
            # Global optimization (no trust region)
            bounds = global_bounds

        # Main loop with rejection for low cosine similarity
        for attempt in range(max_rejection_attempts):
            # Optimize latent using BoTorch acquisition (UCB or LogEI)
            # Use different seed for each attempt to get different candidates
            attempt_seed = iteration * max_rejection_attempts + attempt
            z_opt, log_ei = self.optimize_latent_botorch(
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                verbose=verbose and (attempt == 0),  # Only verbose on first attempt
                seed=attempt_seed,
                bounds=bounds,
                ucb_beta=ucb_beta_override,  # Use adaptive UCB beta if provided
            )

            # Noise injection for diversity (after optimization)
            # Adds small perturbation to encourage exploration of nearby regions
            if noise_scale > 0:
                noise = torch.randn_like(z_opt) * noise_scale
                z_opt = z_opt + noise
                # Clip to valid bounds
                z_opt = torch.clamp(z_opt, bounds[0], bounds[1])
                if verbose and attempt == 0:
                    print(f"  Applied latent noise (scale={noise_scale:.3f})")

            # Denormalize and decode to embedding (32D latent -> 768D embedding)
            # z_opt is normalized [0,1], need to convert back to VAE latent space
            x_range = self.gp.X_max - self.gp.X_min
            z_unnorm = z_opt * x_range + self.gp.X_min

            # Save original z_opt for optimization gap measurement
            z_opt_original = z_opt.clone()

            # Decode to embedding
            self.vae_with_adapter.eval()
            with torch.no_grad():
                z_decode = z_unnorm.to(self.vae_with_adapter.device)
                embedding = self.vae_with_adapter.decode(z_decode)

            # Invert to text
            instruction = self.inverter.invert(embedding.clone())

            if verbose:
                if attempt > 0:
                    print(f"  [Attempt {attempt + 1}] Generated:\n{instruction}")
                else:
                    print(f"  Generated:\n{instruction}")

            # Re-encode for GP prediction
            # IMPORTANT: Predict on GTR(text), not decoder(z), for alignment with training data
            reencoded = self.gtr.encode_tensor(instruction)
            cosine_sim = F.cosine_similarity(
                embedding.unsqueeze(0), reencoded.unsqueeze(0)
            ).item()

            # Check instruction text quality (garbage filtering)
            if not is_valid_instruction(instruction):
                if attempt < max_rejection_attempts - 1:
                    if verbose:
                        print(f"  REJECTED: Invalid instruction (garbage/too short), "
                              f"retrying ({attempt + 1}/{max_rejection_attempts})")
                    continue  # Skip cosine check, go to next attempt
                else:
                    # Log full instruction per CLAUDE.md (never truncate prompts)
                    print(f"WARNING: Accepting invalid instruction after {max_rejection_attempts} attempts")
                    print(f"  Full instruction:\n{instruction}")
                    rejection_attempts = attempt
                    low_quality_accepted = True
                    break  # Accept despite garbage

            # Check cosine similarity threshold
            if cosine_sim >= cosine_sim_threshold:
                # Good alignment, proceed with this candidate
                rejection_attempts = attempt
                low_quality_accepted = False
                break
            else:
                if attempt < max_rejection_attempts - 1:
                    if verbose:
                        print(f"  REJECTED: Cosine sim {cosine_sim:.4f} < {cosine_sim_threshold:.2f}, "
                              f"retrying ({attempt + 1}/{max_rejection_attempts})")
                else:
                    # Always warn about low-quality acceptance - this affects optimization quality
                    print(f"WARNING: Accepting low-quality candidate after {max_rejection_attempts} attempts")
                    print(f"  Cosine similarity: {cosine_sim:.4f} < threshold {cosine_sim_threshold:.2f}")
                    print(f"  This may indicate: Vec2Text inversion issues or threshold too strict")
                    rejection_attempts = attempt
                    low_quality_accepted = True

        # GP prediction on re-encoded embedding (matches training data distribution)
        pred_error, pred_std = self.gp.predict(reencoded)

        # === Optimization Gap Test ===
        # Measure gap between z_opt (BoTorch proposal) and z_real (actual text embedding)
        # If gap is large, GP is optimizing in "empty space" where no real text maps
        with torch.no_grad():
            # Encode re-encoded text back to VAE latent space
            z_real = self.vae_with_adapter.encode_vae(reencoded.unsqueeze(0)).squeeze(0)
            # Normalize to GP input space [0,1] for fair comparison
            denom = self.gp.X_max - self.gp.X_min
            denom[denom == 0] = 1.0
            z_real_norm = (z_real - self.gp.X_min) / denom

            # Gap metrics in VAE latent space (32D) - this is also GP space (no adapter)
            z_opt_z_real_cosine = F.cosine_similarity(
                z_opt_original.unsqueeze(0), z_real_norm.unsqueeze(0)
            ).item()
            z_opt_z_real_euclidean = torch.dist(z_opt_original, z_real_norm).item()

            # GP space is now same as VAE latent space (no adapter compression)
            z_opt_z_real_gp_cosine = z_opt_z_real_cosine  # Same metric, no adapter

            # GP prediction at z_real (actual text) vs at z_opt (dream)
            # This shows if GP was "fooled" by holes in latent space
            pred_error_at_z_real, _ = self.gp.predict(reencoded)

        # Critical warning for large optimization gap (not in summary)
        if z_opt_z_real_cosine < 0.85:
            print(f"  WARNING: Large optimization gap (VAE cosine={z_opt_z_real_cosine:.4f})! "
                  f"GP may be optimizing in empty space.")

        # Evaluate with LLM (or use GP prediction)
        actual_error = None
        if not skip_eval and self.evaluator is not None and self.validation_data is not None:
            prompt = InstructionOnlyPrompt(instruction=instruction, instruction_id=-1)
            actual_error = self.evaluator(prompt, self.validation_data)
            self.total_llm_calls += len(self.validation_data)

        # Update best
        error_to_use = actual_error if actual_error is not None else pred_error
        improved = error_to_use < self.best_error
        if improved:
            self.best_error = error_to_use
            self.best_instruction = instruction

        # === TuRBO: Update trust region state based on iteration result ===
        if self.use_turbo and self.trust_region is not None:
            turbo_info = self.trust_region.update(improved=improved)
            turbo_action = turbo_info["action"]
            trust_region_length = turbo_info["length_after"]

            if verbose and turbo_action != "none":
                if turbo_action == "expand":
                    print(f"  TuRBO: EXPANDED trust region to L={trust_region_length:.4f}")
                elif turbo_action == "shrink":
                    print(f"  TuRBO: SHRUNK trust region to L={trust_region_length:.4f}")
                elif turbo_action == "restart":
                    print(f"  TuRBO: RESTARTED trust region (restart #{turbo_info['restart_count']})")

        # Add observation and retrain GP from scratch
        # Full retraining ensures normalization parameters and noise values are recomputed
        # Determine fidelity for new observation
        if actual_error is not None and self.validation_data is not None:
            # Actual LLM evaluation - use full fidelity
            fidelity = len(self.validation_data)
        elif self.validation_data is not None:
            # GP prediction only - use lower fidelity to reflect model uncertainty
            fidelity = 100  # Conservative fidelity for model predictions
        else:
            fidelity = 100  # Conservative default when no validation data

        self.gp.add_observation(
            reencoded,
            error_to_use,
            fidelity=fidelity,
        )
        retrain_success = self.gp.train(
            epochs=self.config.gp_retrain_epochs,
            patience=self.config.gp_retrain_patience,
            verbose=verbose,
        )
        if not retrain_success:
            self._consecutive_retrain_failures += 1
            self._total_retrain_failures += 1
            print(f"ERROR: GP retraining failed at iteration {iteration}")
            print(f"  Consecutive failures: {self._consecutive_retrain_failures}")
            print(f"  Total failures: {self._total_retrain_failures}")
            print(f"  Training samples: {self.gp.get_training_size()}")

            if self._consecutive_retrain_failures >= 3:
                raise RuntimeError(
                    f"GP retraining failed {self._consecutive_retrain_failures} consecutive times. "
                    f"This indicates a systematic problem with the training data. "
                    f"Possible causes: duplicate observations, numerical overflow, or ill-conditioned kernel."
                )
            if self._total_retrain_failures >= 5:
                print(f"  WARNING: {self._total_retrain_failures} total GP retrain failures - results may be unreliable")
            print(f"  WARNING: Continuing with previous model (attempt {self._consecutive_retrain_failures}/3)")
        else:
            self._consecutive_retrain_failures = 0

        record = IterationRecord(
            iteration=iteration,
            instruction=instruction,
            cosine_similarity=cosine_sim,
            predicted_error=pred_error,
            actual_error=actual_error,
            improved=improved,
            best_error_so_far=self.best_error,
            gp_samples=self.gp.get_training_size(),
            log_ei=log_ei,
            rejection_attempts=rejection_attempts,
            low_quality_accepted=low_quality_accepted,
            # Optimization Gap Test metrics
            z_opt_z_real_cosine=z_opt_z_real_cosine,
            z_opt_z_real_euclidean=z_opt_z_real_euclidean,
            z_opt_z_real_gp_cosine=z_opt_z_real_gp_cosine,
            predicted_error_at_z_real=pred_error_at_z_real,
            # TuRBO trust region state
            trust_region_length=trust_region_length,
            trust_region_action=turbo_action,
            anchor_idx=anchor_idx,
            # Adaptive UCB and noise
            ucb_beta=ucb_beta_override if ucb_beta_override is not None else self.config.ucb_beta,
            noise_applied=noise_scale > 0,
        )

        self.iteration_history.append(record)

        # Track KPI data
        self._predicted_errors.append(pred_error)
        self._actual_errors.append(actual_error)
        self._z_gaps.append(z_opt_z_real_euclidean)

        # Print consolidated iteration summary
        if verbose:
            self._log_iteration_summary(record, pred_std=pred_std)

        # Periodic KPI report every 10 iterations
        if iteration % 10 == 0 and iteration > 0:
            self._log_kpi_report(iteration)

        return record

    def run(
        self,
        iterations: int = 10,
        num_restarts: int = 64,
        raw_samples: int = 1024,
        skip_eval: bool = False,
        verbose: bool = True,
    ) -> List[IterationRecord]:
        """Run multiple optimization iterations.

        Args:
            iterations: Number of iterations
            num_restarts: Number of L-BFGS-B restarts for BoTorch optimization
            raw_samples: Raw samples for initialization seeding
            skip_eval: Skip LLM evaluation
            verbose: Print progress

        Returns:
            List of IterationRecords
        """
        # Check for adaptive UCB beta
        use_adaptive_beta = getattr(self.config, 'ucb_beta_adaptive', True)
        beta_init = self.config.ucb_beta
        beta_final = getattr(self.config, 'ucb_beta_final', 2.0)
        noise_scale = getattr(self.config, 'latent_noise_scale', 0.05)

        if verbose:
            print("\n" + "=" * 60)
            print("Starting LIPO Inference (BoTorch qLogEI)")
            print("=" * 60)
            print(f"  Initial best error: {self.best_error:.4f}")
            print(f"  GP samples: {self.gp.get_training_size()}")
            print(f"  BoTorch: {num_restarts} restarts, {raw_samples} raw samples")
            if use_adaptive_beta and self.config.acquisition_type == "ucb":
                print(f"  Adaptive UCB β: {beta_init:.1f} → {beta_final:.1f} over {iterations} iterations")
            if noise_scale > 0:
                print(f"  Latent noise injection: scale={noise_scale:.3f}")

        for i in range(iterations):
            # Compute adaptive UCB beta (linear decay)
            if use_adaptive_beta:
                progress = i / max(iterations - 1, 1)
                current_ucb_beta = beta_init * (1 - progress) + beta_final * progress
            else:
                current_ucb_beta = beta_init

            self.run_iteration(
                iteration=i + 1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                skip_eval=skip_eval,
                verbose=verbose,
                total_iterations=iterations,
                ucb_beta_override=current_ucb_beta,
                noise_scale=noise_scale,
            )

        if verbose:
            print("\n" + "=" * 60)
            print("LIPO Inference Complete")
            print("=" * 60)
            print(f"  Best error: {self.best_error:.4f}")
            print(f"  Best instruction:\n{self.best_instruction}")
            print(f"  Total LLM calls: {self.total_llm_calls}")

        return self.iteration_history

    def round_trip_test(self, instruction: str) -> dict:
        """Test reconstruction fidelity: Text → GTR → VAE → Vec2Text → Text'

        This measures how well the VAE + Vec2Text pipeline can reconstruct
        a known instruction. High similarity (>0.95) means good reconstruction.

        Args:
            instruction: Original instruction text to test

        Returns:
            dict with:
                - semantic_similarity: cosine similarity between GTR(original) and GTR(reconstructed)
                - reconstructed_text: the reconstructed instruction
                - original_text: the input instruction
        """
        # Encode original text
        emb_original = self.gtr.encode_tensor(instruction)

        # VAE encode -> decode (full round-trip through latent space)
        # Ensure device consistency for cross-device scenarios
        self.vae_with_adapter.eval()
        vae_dev = self.vae_with_adapter.device
        with torch.no_grad():
            emb_for_vae = emb_original.to(vae_dev)
            z = self.vae_with_adapter.encode_vae(emb_for_vae.unsqueeze(0))
            emb_decoded = self.vae_with_adapter.decode(z.squeeze(0))

        # Vec2Text inversion (embedding -> text)
        reconstructed = self.inverter.invert(emb_decoded.clone())

        # Re-encode reconstructed text
        emb_reconstructed = self.gtr.encode_tensor(reconstructed)

        # Compute semantic similarity in GTR embedding space
        semantic_sim = F.cosine_similarity(
            emb_original.unsqueeze(0),
            emb_reconstructed.unsqueeze(0)
        ).item()

        return {
            "semantic_similarity": semantic_sim,
            "reconstructed_text": reconstructed,
            "original_text": instruction,
        }

    def run_round_trip_diagnostic(
        self, instructions: List[str], verbose: bool = True
    ) -> dict:
        """Run round-trip test on multiple instructions.

        Recommended: Run on top-K training instructions before inference
        to establish baseline reconstruction quality.

        Args:
            instructions: List of instructions to test
            verbose: Print summary

        Returns:
            dict with:
                - mean_similarity: average cosine similarity
                - min_similarity: worst case
                - max_similarity: best case
                - below_90: count of instructions with sim < 0.90 (poor)
                - below_95: count of instructions with sim < 0.95 (acceptable)
                - results: list of individual test results
        """
        import numpy as np

        results = []
        for inst in instructions:
            result = self.round_trip_test(inst)
            results.append(result)

        sims = [r["semantic_similarity"] for r in results]
        summary = {
            "mean_similarity": float(np.mean(sims)),
            "min_similarity": float(np.min(sims)),
            "max_similarity": float(np.max(sims)),
            "below_90": sum(1 for s in sims if s < 0.90),
            "below_95": sum(1 for s in sims if s < 0.95),
            "results": results,
        }

        if verbose:
            print(f"\n{'=' * 60}")
            print("ROUND-TRIP DIAGNOSTIC")
            print("=" * 60)
            print(f"Tested: {len(instructions)} instructions")
            print(f"Mean similarity: {summary['mean_similarity']:.4f}")
            print(f"Min: {summary['min_similarity']:.4f}, Max: {summary['max_similarity']:.4f}")
            print(f"Below 0.90 (poor): {summary['below_90']}")
            print(f"Below 0.95 (acceptable): {summary['below_95']}")

            # Interpretation
            if summary['mean_similarity'] >= 0.95:
                print("Interpretation: EXCELLENT - VAE+Vec2Text preserve meaning well")
            elif summary['mean_similarity'] >= 0.90:
                print("Interpretation: ACCEPTABLE - minor semantic drift expected")
            else:
                print("Interpretation: POOR - significant meaning loss in reconstruction")
                print("  This may explain lack of optimization improvement.")

            # Show worst case if it's particularly bad
            if summary['min_similarity'] < 0.85:
                worst_idx = sims.index(summary['min_similarity'])
                worst = results[worst_idx]
                print(f"\nWorst reconstruction (sim={worst['semantic_similarity']:.4f}):")
                print(f"  Original: {worst['original_text'][:100]}...")
                print(f"  Reconstructed: {worst['reconstructed_text'][:100]}...")

        return summary
