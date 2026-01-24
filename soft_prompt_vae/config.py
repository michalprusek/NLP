"""Centralized configuration for Soft-Prompt VAE.

All hyperparameters and settings are defined here using dataclasses
for type safety and easy serialization.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Base model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    torch_dtype: str = "bfloat16"
    # device_map is set dynamically in model.py based on DDP vs single GPU

    # VAE latent space
    latent_dim: int = 64
    hidden_dim: int = 2048
    llama_hidden_size: int = 4096  # Llama-3.1-8B hidden size

    # Soft prompt
    num_soft_tokens: int = 32

    # ========== Matryoshka Representation Learning ==========
    # Nested latent dimensions for coarse-to-fine LSBO optimization
    # When set (e.g., (16, 32, 64)), training randomly samples active_dim
    # and masks higher dimensions, forcing hierarchical structure
    matryoshka_dims: Optional[Tuple[int, ...]] = None  # e.g., (16, 32, 64)
    full_dim_probability: float = 0.33  # Probability of using full dimension during training

    # LoRA configuration
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    use_rslora: bool = True  # Rank-Stabilized LoRA

    # Attention pooling
    num_attention_heads: int = 8

    # Word dropout for decoder (combats posterior collapse)
    word_dropout_rate: float = 0.4  # Randomly drop 40% of decoder input tokens

    # Bag-of-Words auxiliary loss (forces semantic content into z)
    bow_loss_weight: float = 1.5  # Weight for BoW multi-label classification loss

    # Deep prefix tuning (inject z into all attention layers via past_key_values)
    # NOTE: Deep prefix is INCOMPATIBLE with gradient checkpointing due to memory
    # When disabled, gradient checkpointing saves ~20GB VRAM on Llama-8B
    # CDP-VAE: Disabled by default for memory efficiency; enable for stronger conditioning
    use_deep_prefix: bool = False  # When True, uses past_key_values instead of input embeddings
    deep_prefix_layers: Optional[int] = 8  # Number of TOP layers to inject (None = all layers, reduces memory)
    deep_prefix_bottleneck: int = 128  # Bottleneck dimension for memory-efficient deep prefix

    # ========== CDP-VAE Contrastive Learning ==========
    # InfoNCE contrastive loss prevents encoder from mapping all inputs to same z
    contrastive_weight: float = 0.1  # Weight for InfoNCE loss (0 to disable)
    contrastive_temperature: float = 0.07  # Temperature for InfoNCE (lower = more discriminative)

    # ========== Text Augmentation ==========
    # Token-level augmentation creates positive pairs for contrastive learning
    augmentation_span_mask_prob: float = 0.15  # Probability of masking spans
    augmentation_word_dropout: float = 0.1  # Token dropout rate for augmentation
    augmentation_probability: float = 0.5  # Probability of computing augmented encoding (memory optimization)

    def __post_init__(self):
        """Validate configuration."""
        assert self.latent_dim > 0, "latent_dim must be positive"
        assert self.num_soft_tokens > 0, "num_soft_tokens must be positive"
        assert self.lora_r > 0, "lora_r must be positive"
        assert 0.0 <= self.contrastive_weight <= 1.0, "contrastive_weight must be in [0, 1]"
        assert 0.0 < self.contrastive_temperature <= 1.0, "contrastive_temperature must be in (0, 1]"

        # Matryoshka validation
        if self.matryoshka_dims is not None:
            assert len(self.matryoshka_dims) >= 2, "matryoshka_dims must have at least 2 levels"
            assert self.matryoshka_dims[-1] == self.latent_dim, (
                f"Last matryoshka_dim ({self.matryoshka_dims[-1]}) must equal latent_dim ({self.latent_dim})"
            )
            assert self.matryoshka_dims == tuple(sorted(self.matryoshka_dims)), (
                "matryoshka_dims must be in ascending order"
            )
            assert all(d > 0 for d in self.matryoshka_dims), "All matryoshka_dims must be positive"
            assert 0.0 <= self.full_dim_probability <= 1.0, "full_dim_probability must be in [0, 1]"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Optimization
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Batch sizes for 2x L40S (48GB each)
    # NOTE: Small batch_size=8 is intentional for VAE training due to:
    # - Llama-8B model weights (~16GB)
    # - LoRA adapter gradients
    # - VAE encoder/decoder overhead
    # - BoW vocabulary logits (vocab_size * batch)
    # - Contrastive learning similarity matrices
    # Profile with larger values if VRAM allows
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 12
    # Effective batch = 8 * 12 * 2 GPUs = 192

    # Training duration
    num_epochs: int = 10
    max_steps: Optional[int] = None

    # KL annealing (cyclical)
    beta_max: float = 1.0
    num_cycles: int = 4
    cycle_ratio: float = 0.5  # 50% ramp up, 50% hold

    # Free bits (minimum KL per dimension)
    free_bits: float = 0.5

    # Checkpointing
    save_steps: int = 100
    eval_steps: int = 500
    logging_steps: int = 100

    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"

    # ========== CDP-VAE Collapse Monitoring ==========
    # Adaptive intervention to prevent/recover from posterior collapse
    enable_collapse_monitoring: bool = True  # Enable collapse detection and intervention
    collapse_au_warning: float = 0.3  # Active Unit ratio warning threshold (level 1)
    collapse_au_critical: float = 0.1  # Active Unit ratio critical threshold (level 3)

    # Paths
    output_dir: Path = field(default_factory=lambda: Path("soft_prompt_vae/checkpoints"))
    logging_dir: Path = field(default_factory=lambda: Path("soft_prompt_vae/logs"))

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size assuming 2 GPUs."""
        return self.per_device_batch_size * self.gradient_accumulation_steps * 2


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    # Tokenization
    max_instruction_length: int = 512
    max_response_length: int = 1024
    min_instruction_length: int = 10
    min_response_length: int = 20

    # Language filtering
    language: str = "en"
    language_threshold: float = 0.8

    # Deduplication
    minhash_threshold: float = 0.85
    minhash_num_perm: int = 128
    minhash_ngram: int = 5

    # DataLoader
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2

    # Paths
    processed_dir: Path = field(default_factory=lambda: Path("soft_prompt_vae/processed"))
    fasttext_model: Path = field(
        default_factory=lambda: Path("soft_prompt_vae/data/lid.176.bin")
    )

    # Dataset phases (for incremental training)
    phase_1_datasets: Tuple[str, ...] = ("mlabonne/FineTome-100k",)
    phase_2_datasets: Tuple[str, ...] = (
        "teknium/OpenHermes-2.5",
        "WizardLMTeam/WizardLM_evol_instruct_V2_196k",
    )
    phase_3_datasets: Tuple[str, ...] = (
        "HuggingFaceH4/no_robots",
        "ise-uiuc/Magicoder-OSS-Instruct-75K",
    )


@dataclass
class VAEConfig:
    """Combined configuration for the full VAE system."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def save(self, path: Path) -> None:
        """Save configuration to JSON."""
        import json
        from dataclasses import asdict

        def convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=convert)

    @classmethod
    def load(cls, path: Path) -> "VAEConfig":
        """Load configuration from JSON."""
        import json

        with open(path) as f:
            data = json.load(f)

        # Convert paths back
        for section in ["training", "data"]:
            if section in data:
                for key, value in data[section].items():
                    if key.endswith("_dir") or key.endswith("_model"):
                        data[section][key] = Path(value)

        return cls(
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            data=DataConfig(**data.get("data", {})),
        )


# Default configuration instance
DEFAULT_CONFIG = VAEConfig()
