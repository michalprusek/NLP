"""Soft-Prompt VAE with Llama-3.1-8B backbone.

Architecture:
- Encoder: Llama hidden states → Attention Pooling → MLP → μ, logσ²
- Latent: 64D Gaussian with reparameterization
- Decoder: z → MLP → 32 soft prompt tokens → Llama generation
"""

import logging
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from soft_prompt_vae.augmentation import TextAugmenter
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import DynamicCache
from peft import LoraConfig, get_peft_model, TaskType

from soft_prompt_vae.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class VAEOutput:
    """Output from VAE forward pass."""

    # Reconstruction
    logits: torch.Tensor  # (batch, seq_len, vocab_size)
    loss: Optional[torch.Tensor] = None

    # Latent space
    z: Optional[torch.Tensor] = None  # (batch, latent_dim)
    mu: Optional[torch.Tensor] = None  # (batch, latent_dim)
    logvar: Optional[torch.Tensor] = None  # (batch, latent_dim)

    # For analysis
    kl_loss: Optional[torch.Tensor] = None
    recon_loss: Optional[torch.Tensor] = None

    # Bag-of-Words auxiliary output
    bow_logits: Optional[torch.Tensor] = None  # (batch, vocab_size)

    # Contrastive learning (CDP-VAE)
    mu_augmented: Optional[torch.Tensor] = None  # (batch, latent_dim) - augmented view for InfoNCE

    # Matryoshka representation learning
    active_matryoshka_dim: Optional[int] = None  # Currently active dimension (e.g., 16, 32, or 64)


class AttentionPooling(nn.Module):
    """Learnable attention pooling over sequence dimension.

    Converts (batch, seq_len, hidden_dim) → (batch, hidden_dim)
    using multi-head self-attention with a learnable query.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Learnable query token (scaled for proper attention initialization)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * (hidden_dim ** -0.5))

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool sequence into single vector.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len) - 1 for valid, 0 for padding

        Returns:
            Pooled representation (batch, hidden_dim)
        """
        batch_size = hidden_states.size(0)

        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)

        # Convert attention mask to key_padding_mask
        # MultiheadAttention expects True for positions to ignore
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        # Attend over sequence
        pooled, _ = self.attention(
            query=query,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
        )

        # Remove sequence dimension and normalize
        pooled = pooled.squeeze(1)
        pooled = self.layer_norm(pooled)

        return pooled


class VariationalEncoder(nn.Module):
    """Variational encoder: hidden_dim → latent distribution.

    MLP that maps pooled representation to μ and log(σ²).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # MLP: input_dim → hidden_dim → hidden_dim → 2*latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Separate heads for μ and logvar
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # Initialize logvar head with small weights for stable variance learning
        # CRITICAL: Do NOT use zeros - that prevents gradient flow through inputs
        nn.init.normal_(self.logvar_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.logvar_head.bias, -2.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution.

        Args:
            x: Pooled representation (batch, input_dim)

        Returns:
            mu: Mean (batch, latent_dim)
            logvar: Log variance (batch, latent_dim)
        """
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ * ε.

        Args:
            mu: Mean (batch, latent_dim)
            logvar: Log variance (batch, latent_dim)
            training: Whether to add noise (False for inference)

        Returns:
            z: Sampled latent (batch, latent_dim)
        """
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu


class SoftPromptProjector(nn.Module):
    """Project latent z to soft prompt tokens.

    Maps z → num_soft_tokens × hidden_dim embeddings.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        llama_hidden_size: int,
        num_soft_tokens: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_soft_tokens = num_soft_tokens
        self.llama_hidden_size = llama_hidden_size

        # MLP: latent_dim → hidden_dim → llama_hidden_size → num_tokens * llama_hidden_size
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, llama_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llama_hidden_size, num_soft_tokens * llama_hidden_size),
        )

        # Learnable position embeddings for soft tokens
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_soft_tokens, llama_hidden_size) * 0.02
        )

        # Layer norm for output
        self.layer_norm = nn.LayerNorm(llama_hidden_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Project latent to soft prompt tokens.

        Args:
            z: Latent vector (batch, latent_dim)

        Returns:
            Soft prompt embeddings (batch, num_soft_tokens, llama_hidden_size)
        """
        batch_size = z.size(0)

        # Project to flat representation
        flat = self.projector(z)

        # Reshape to sequence
        soft_tokens = flat.view(batch_size, self.num_soft_tokens, self.llama_hidden_size)

        # Add position embeddings
        soft_tokens = soft_tokens + self.position_embeddings

        # Normalize
        soft_tokens = self.layer_norm(soft_tokens)

        return soft_tokens


class DeepPrefixProjector(nn.Module):
    """Project latent z to past_key_values for Llama layers.

    Memory-optimized design:
    1. Fused K+V projection per layer (reduces memory vs separate projections)
    2. Smaller bottleneck (128) and hidden_dim (1024) for memory efficiency
    3. No bias in per-layer projections (small memory savings)
    4. Contiguous output tensors for efficient CUDA operations

    Note: The main memory cost with deep prefix is NOT this projector, but rather
    the fact that past_key_values disables gradient checkpointing in transformers.
    Without grad checkpointing, all intermediate activations must be stored.
    """

    def __init__(
        self,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        prefix_len: int,
        total_model_layers: int = 32,
        hidden_dim: int = 1024,  # Reduced from 2048 for memory
        bottleneck_dim: int = 128,  # Reduced from 256 for memory
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.prefix_len = prefix_len
        self.kv_dim = num_heads * head_dim
        self.bottleneck_dim = bottleneck_dim

        # Shared projection: z -> shared representation for all layers
        self.shared_projector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, prefix_len * bottleneck_dim),
        )

        # Fused K+V projections per layer (no bias for memory efficiency)
        self.kv_projections = nn.ModuleList([
            nn.Linear(bottleneck_dim, 2 * self.kv_dim, bias=False)
            for _ in range(num_layers)
        ])

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(bottleneck_dim)

        # Learnable per-layer scales
        self.layer_scales = nn.Parameter(torch.ones(num_layers) * 0.1)

    def forward(self, z: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """Project latent to past_key_values format.

        Args:
            z: Latent vector (batch, latent_dim)

        Returns:
            Tuple of (key, value) pairs for each layer.
            Each key/value has shape (batch, num_heads, prefix_len, head_dim)
        """
        batch_size = z.size(0)

        # Shared projection
        shared = self.shared_projector(z)
        shared = shared.view(batch_size, self.prefix_len, self.bottleneck_dim)
        shared = self.layer_norm(shared)

        past_key_values = []
        for layer_idx in range(self.num_layers):
            # Fused K+V projection with layer-specific scaling
            kv = self.kv_projections[layer_idx](shared) * self.layer_scales[layer_idx]
            key, value = kv.chunk(2, dim=-1)

            # Reshape to (batch, num_heads, prefix_len, head_dim)
            key = key.view(batch_size, self.prefix_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
            value = value.view(batch_size, self.prefix_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

            past_key_values.append((key, value))

        return tuple(past_key_values)


class LlamaSoftPromptVAE(nn.Module):
    """Full VAE model with Llama-3.1-8B backbone.

    Encoder: Instruction → Llama → Attention Pool → Variational Encoder → z
    Decoder: z → Soft Prompts → Llama → Response reconstruction
    """

    def __init__(self, config: ModelConfig, use_ddp: bool = True):
        super().__init__()
        self.config = config

        logger.info(f"Loading Llama model: {config.model_name}")

        # Load Llama model
        # For DDP: device_map=None, Accelerator handles placement
        # For single GPU with offloading: device_map="auto"
        # For single GPU without offloading: device_map={"": 0}
        if use_ddp:
            device_map = None
        else:
            # Use explicit GPU placement for speed (no memory offloading)
            device_map = {"": 0}

        self.llama = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=getattr(torch, config.torch_dtype),
            device_map=device_map,
            trust_remote_code=True,
        )

        # Apply LoRA
        self._apply_lora()

        # Freeze base model, only LoRA adapters are trainable
        for name, param in self.llama.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False

        # Get model dtype for VAE components
        model_dtype = getattr(torch, config.torch_dtype)

        # VAE components (same dtype as LLM)
        self.attention_pooling = AttentionPooling(
            hidden_dim=config.llama_hidden_size,
            num_heads=config.num_attention_heads,
        ).to(model_dtype)

        self.variational_encoder = VariationalEncoder(
            input_dim=config.llama_hidden_size,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
        ).to(model_dtype)

        self.soft_prompt_projector = SoftPromptProjector(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            llama_hidden_size=config.llama_hidden_size,
            num_soft_tokens=config.num_soft_tokens,
        ).to(model_dtype)

        # Bag-of-Words head for auxiliary loss (predicts which tokens appear in output)
        vocab_size = self.llama.config.vocab_size
        self.bow_head = nn.Linear(config.latent_dim, vocab_size).to(model_dtype)

        # Deep prefix projector (optional, for past_key_values injection)
        # WARNING: Deep prefix DISABLES gradient checkpointing, requiring ~2x more VRAM
        self.deep_prefix_projector = None
        if config.use_deep_prefix:
            llama_config = self.llama.config
            num_layers = llama_config.num_hidden_layers
            self.deep_prefix_projector = DeepPrefixProjector(
                latent_dim=config.latent_dim,
                num_layers=num_layers,
                num_heads=llama_config.num_key_value_heads,  # Use KV heads for GQA
                head_dim=llama_config.hidden_size // llama_config.num_attention_heads,
                prefix_len=config.num_soft_tokens,
                hidden_dim=config.hidden_dim,
                bottleneck_dim=config.deep_prefix_bottleneck,
            ).to(model_dtype)
            logger.info(
                f"Deep prefix projector enabled: {num_layers} layers, "
                f"{config.num_soft_tokens} prefix tokens, bottleneck={config.deep_prefix_bottleneck}"
            )
            logger.warning(
                "Deep prefix DISABLES gradient checkpointing. "
                "Ensure you have sufficient VRAM (~48GB+ for Llama-8B)."
            )

        # Matryoshka masker for hierarchical latent dimensions
        self.matryoshka_masker = None
        if config.matryoshka_dims is not None:
            from soft_prompt_vae.matryoshka import MatryoshkaMasker
            self.matryoshka_masker = MatryoshkaMasker(
                nested_dims=config.matryoshka_dims,
                full_dim_probability=config.full_dim_probability,
            )
            logger.info(
                f"Matryoshka masker enabled: dims={config.matryoshka_dims}, "
                f"full_dim_prob={config.full_dim_probability}"
            )

        logger.info(
            f"VAE initialized: latent_dim={config.latent_dim}, "
            f"num_soft_tokens={config.num_soft_tokens}, dtype={model_dtype}, "
            f"bow_head=True, deep_prefix={config.use_deep_prefix}, "
            f"matryoshka={config.matryoshka_dims is not None}"
        )

    def _apply_lora(self) -> None:
        """Apply LoRA adapters to Llama."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.lora_target_modules),
            use_rslora=self.config.use_rslora,
        )

        self.llama = get_peft_model(self.llama, lora_config)

        # Log trainable parameters
        trainable_params = sum(
            p.numel() for p in self.llama.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.llama.parameters())
        logger.info(
            f"LoRA applied: {trainable_params:,} trainable / "
            f"{total_params:,} total ({100 * trainable_params / total_params:.2f}%)"
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        force_matryoshka_dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[int]]:
        """Encode instruction to latent distribution.

        Args:
            input_ids: Instruction token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            force_matryoshka_dim: Force a specific Matryoshka dimension (None = random during training)

        Returns:
            z: Sampled latent (batch, latent_dim)
            mu: Mean (batch, latent_dim)
            logvar: Log variance (batch, latent_dim)
            active_dim: Active Matryoshka dimension (None if Matryoshka disabled)
        """
        # Get Llama hidden states
        # For PEFT models, access the underlying transformer model
        base_model = self.llama.base_model.model.model  # PeftModel -> base -> LlamaForCausalLM -> LlamaModel
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Use last hidden state
        hidden_states = outputs.last_hidden_state

        # Pool to single vector
        pooled = self.attention_pooling(hidden_states, attention_mask)

        # Encode to latent distribution
        mu, logvar = self.variational_encoder(pooled)

        # Apply Matryoshka masking if enabled
        active_dim = None
        if self.matryoshka_masker is not None:
            matryoshka_out = self.matryoshka_masker(
                mu, logvar,
                training=self.training,
                force_dim=force_matryoshka_dim,
            )
            mu = matryoshka_out.mu_masked
            logvar = matryoshka_out.logvar_masked
            active_dim = matryoshka_out.active_dim

        # Sample z (using potentially masked mu/logvar)
        z = self.variational_encoder.reparameterize(mu, logvar, self.training)

        return z, mu, logvar, active_dim

    def encode_with_augmentation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        augmenter: Optional["TextAugmenter"] = None,
        augmentation_prob: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[int]]:
        """Encode with optional augmentation for contrastive learning.

        Args:
            input_ids: Instruction token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            augmenter: Optional TextAugmenter instance for creating positive pairs
            augmentation_prob: Probability of computing augmented encoding

        Returns:
            z: Sampled latent (batch, latent_dim)
            mu: Mean (batch, latent_dim)
            logvar: Log variance (batch, latent_dim)
            mu_augmented: Augmented mean (batch, latent_dim) or None
            active_dim: Active Matryoshka dimension (None if Matryoshka disabled)
        """
        z, mu, logvar, active_dim = self.encode(input_ids, attention_mask)

        mu_augmented = None
        if self.training and augmenter is not None and torch.rand(1).item() < augmentation_prob:
            aug_ids, aug_mask = augmenter.augment(input_ids, attention_mask)
            with torch.no_grad():
                # IMPORTANT: Use same active_dim for augmented view to ensure
                # contrastive learning compares like-with-like
                _, mu_augmented, _, _ = self.encode(
                    aug_ids, aug_mask,
                    force_matryoshka_dim=active_dim,
                )

        return z, mu, logvar, mu_augmented, active_dim

    def apply_word_dropout(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        dropout_rate: float,
    ) -> torch.Tensor:
        """Apply word dropout to input tokens during training.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            dropout_rate: Probability of dropping each token

        Returns:
            Token IDs with some replaced by pad_token_id
        """
        if not self.training or dropout_rate <= 0:
            return input_ids

        dropout_mask = torch.bernoulli(
            torch.full_like(input_ids, 1.0 - dropout_rate, dtype=torch.float)
        ).bool()
        dropout_mask[:, 0] = True  # Preserve BOS token

        pad_token_id = self.llama.config.pad_token_id or 0
        return torch.where(dropout_mask, input_ids, pad_token_id)

    def decode(
        self,
        z: torch.Tensor,
        target_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent to response logits.

        Supports two modes:
        1. Standard: Soft prompts concatenated with input embeddings
        2. Deep Prefix: z injected into all layers via past_key_values

        Args:
            z: Latent vector (batch, latent_dim)
            target_ids: Target token IDs (batch, seq_len)
            target_attention_mask: Target attention mask (batch, seq_len)

        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        batch_size = z.size(0)

        # Apply word dropout during training to combat posterior collapse
        if self.training and self.config.word_dropout_rate > 0:
            target_ids = self.apply_word_dropout(
                target_ids, target_attention_mask, self.config.word_dropout_rate
            )

        # Get target embeddings (navigate PEFT model structure)
        embed_tokens = self.llama.base_model.model.model.embed_tokens
        target_embeds = embed_tokens(target_ids)

        # Create prefix mask (used by both modes)
        prefix_mask = torch.ones(
            batch_size, self.config.num_soft_tokens,
            dtype=target_attention_mask.dtype,
            device=target_attention_mask.device,
        )
        full_attention_mask = torch.cat([prefix_mask, target_attention_mask], dim=1)

        if self.deep_prefix_projector is not None:
            # Deep Prefix Mode: Inject z into attention layers via past_key_values
            past_key_values_tuple = self.deep_prefix_projector(z)
            cache = DynamicCache()
            for layer_idx, (key, value) in enumerate(past_key_values_tuple):
                cache.update(key, value, layer_idx)

            outputs = self.llama(
                inputs_embeds=target_embeds,
                attention_mask=full_attention_mask,
                past_key_values=cache,
                return_dict=True,
            )
            logits = outputs.logits

        else:
            # Standard Mode: Concatenate soft prompts with input embeddings
            soft_prompts = self.soft_prompt_projector(z)
            inputs_embeds = torch.cat([soft_prompts, target_embeds], dim=1)

            outputs = self.llama(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                return_dict=True,
            )
            # Extract logits for target positions (skip soft prompt positions)
            logits = outputs.logits[:, self.config.num_soft_tokens:, :]

        return logits

    def forward(
        self,
        instruction_ids: torch.Tensor,
        instruction_attention_mask: torch.Tensor,
        response_ids: torch.Tensor,
        response_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        augmenter: Optional["TextAugmenter"] = None,
    ) -> VAEOutput:
        """Full forward pass.

        Args:
            instruction_ids: Instruction token IDs (batch, instr_len)
            instruction_attention_mask: Instruction attention mask
            response_ids: Response token IDs (batch, resp_len)
            response_attention_mask: Response attention mask
            labels: Labels for loss computation (-100 for ignored)
            augmenter: Optional TextAugmenter for contrastive learning (CDP-VAE)

        Returns:
            VAEOutput with logits, loss, latent info, bow_logits, and mu_augmented
        """
        # Encode instruction to latent (with optional augmentation for contrastive loss)
        z, mu, logvar, mu_augmented, active_matryoshka_dim = self.encode_with_augmentation(
            instruction_ids, instruction_attention_mask, augmenter,
            augmentation_prob=self.config.augmentation_probability,
        )

        # Decode latent to response logits
        logits = self.decode(z, response_ids, response_attention_mask)

        # Compute Bag-of-Words logits (for auxiliary loss)
        bow_logits = self.bow_head(z)  # (batch, vocab_size)

        # Compute loss if labels provided
        loss = None
        recon_loss = None
        kl_loss = None

        if labels is not None:
            # Reconstruction loss (cross-entropy)
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            recon_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="mean",
            )

            # KL divergence (will be weighted by beta in loss function)
            kl_loss = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp()
            )

            # Total loss (beta=1.0 placeholder, actual annealing in loss.py)
            loss = recon_loss + kl_loss

        return VAEOutput(
            logits=logits,
            loss=loss,
            z=z,
            mu=mu,
            logvar=logvar,
            kl_loss=kl_loss,
            recon_loss=recon_loss,
            bow_logits=bow_logits,
            mu_augmented=mu_augmented,
            active_matryoshka_dim=active_matryoshka_dim,
        )

    def generate(
        self,
        z: torch.Tensor,
        max_length: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Generate response from latent.

        Args:
            z: Latent vector (batch, latent_dim)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample (vs greedy)

        Returns:
            Generated token IDs (batch, gen_len)
        """
        batch_size = z.size(0)
        device = z.device

        # Generate soft prompts
        soft_prompts = self.soft_prompt_projector(z)

        # Start with BOS token (use pad token as BOS for Llama)
        bos_id = self.llama.config.bos_token_id or self.llama.config.pad_token_id
        input_ids = torch.full(
            (batch_size, 1), bos_id, dtype=torch.long, device=device
        )

        # Generate autoregressively
        embed_tokens = self.llama.base_model.model.model.embed_tokens
        generated = self.llama.generate(
            inputs_embeds=torch.cat([
                soft_prompts,
                embed_tokens(input_ids),
            ], dim=1),
            max_length=max_length + self.config.num_soft_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.llama.config.pad_token_id,
            eos_token_id=self.llama.config.eos_token_id,
        )

        # Remove soft prompt positions from output
        return generated[:, self.config.num_soft_tokens:]

    def interpolate(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Interpolate between two latent points.

        Args:
            z1: Start latent (1, latent_dim)
            z2: End latent (1, latent_dim)
            num_steps: Number of interpolation steps

        Returns:
            Interpolated latents (num_steps, latent_dim)
        """
        alphas = torch.linspace(0, 1, num_steps, device=z1.device)
        interpolated = torch.stack([
            (1 - alpha) * z1 + alpha * z2 for alpha in alphas
        ]).squeeze(1)
        return interpolated

    def get_trainable_parameters(self) -> int:
        """Get count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: ModelConfig) -> LlamaSoftPromptVAE:
    """Create VAE model from config.

    Args:
        config: Model configuration

    Returns:
        LlamaSoftPromptVAE instance
    """
    return LlamaSoftPromptVAE(config)
