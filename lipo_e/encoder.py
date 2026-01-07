"""Encoders for LIPO-E.

Components:
- GTREncoder: Sentence-Transformers GTR-T5-Base (768D, L2-normalized)
- InstructionEncoder: 768D → 16D VAE encoder
- StructureAwareVAE: Combined instruction + exemplar VAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from sentence_transformers import SentenceTransformer

from lipo_e.set_transformer import ExemplarSetEncoder


class GTREncoder:
    """GTR-T5-Base encoder for instruction and exemplar embeddings.

    Produces L2-normalized 768D embeddings compatible with Vec2Text.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = SentenceTransformer(
            "sentence-transformers/gtr-t5-base",
            device=device,
        )
        self.model.eval()

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings.

        Args:
            texts: List of strings to encode

        Returns:
            embeddings: (len(texts), 768) L2-normalized
        """
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return embeddings.to(self.device)

    def encode_single(self, text: str) -> torch.Tensor:
        """Encode single text."""
        return self.encode([text])[0]


class InstructionEncoder(nn.Module):
    """VAE encoder for instruction embeddings.

    Architecture:
        768 → 256 → 128 → 32 (μ + logvar) → 16D latent
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        latent_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim * 2),  # μ + logvar
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Instruction embedding (batch, 768)

        Returns:
            mu, logvar: (batch, latent_dim)
        """
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        # Clamp logvar for stability
        logvar = torch.clamp(logvar, min=-10, max=2)
        return mu, logvar


class InstructionDecoder(nn.Module):
    """VAE decoder for instruction embeddings.

    Architecture:
        16D → 128 → 256 → 768 (L2-normalized)
    """

    def __init__(
        self,
        latent_dim: int = 16,
        embedding_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent (batch, latent_dim)

        Returns:
            embedding: L2-normalized (batch, 768)
        """
        x = self.decoder(z)
        return F.normalize(x, p=2, dim=-1)


class SlotBasedExemplarDecoder(nn.Module):
    """Decode exemplar latent to soft selection over exemplar pool.

    Uses learned slot queries to attend to candidate exemplars.
    Produces soft weights during training (Gumbel-Softmax),
    hard selection during inference (argmax).

    Architecture:
        z_exemplar (16D) → MLP → slot modulation
        Slot queries (8) × Pool embeddings (N) → selection logits
        + num_exemplars prediction head (0-8)
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        num_slots: int = 8,
        pool_embedding_dim: int = 768,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.latent_dim = latent_dim

        # Latent to slot query modulation
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),  # 32D
        )

        # Learned slot bases
        slot_dim = hidden_dim // 2  # 32D
        self.slot_bases = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.02)

        # Pool embedding projection
        self.pool_proj = nn.Linear(pool_embedding_dim, slot_dim)

        # Selection scoring (dot product + learned bias)
        self.selection_bias = nn.Parameter(torch.zeros(num_slots, 1))

        # Number of exemplars prediction
        self.num_exemplars_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, num_slots + 1),  # 0 to num_slots
        )

    def forward(
        self,
        z: torch.Tensor,
        pool_embeddings: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Exemplar latent (batch, latent_dim)
            pool_embeddings: All candidate exemplars (N_pool, 768)
            temperature: Gumbel-Softmax temperature
            hard: Use hard selection (inference mode)

        Returns:
            selection_probs: (batch, num_slots, N_pool)
            num_ex_logits: (batch, num_slots + 1)
            selected_indices: (batch, num_slots)
        """
        batch_size = z.shape[0]
        N_pool = pool_embeddings.shape[0]

        # Project latent to slot modulation
        latent_mod = self.latent_proj(z)  # (batch, 32)

        # Create slot queries: base + latent modulation
        # Each slot gets same modulation but different base
        slot_queries = self.slot_bases.unsqueeze(0) + latent_mod.unsqueeze(1)
        # (batch, num_slots, 32)

        # Project pool embeddings
        pool_features = self.pool_proj(pool_embeddings)  # (N_pool, 32)

        # Compute selection logits via dot product
        # slot_queries: (batch, num_slots, 32)
        # pool_features: (N_pool, 32)
        selection_logits = torch.matmul(
            slot_queries,  # (batch, num_slots, 32)
            pool_features.T,  # (32, N_pool)
        )  # (batch, num_slots, N_pool)

        # Add learned bias
        selection_logits = selection_logits + self.selection_bias

        # Apply temperature-scaled softmax or Gumbel-Softmax
        if hard:
            # Hard selection via argmax
            selected_indices = selection_logits.argmax(dim=-1)  # (batch, num_slots)
            selection_probs = F.one_hot(selected_indices, N_pool).float()
        else:
            # Soft selection via Gumbel-Softmax
            selection_probs = F.gumbel_softmax(
                selection_logits,
                tau=temperature,
                hard=False,
                dim=-1,
            )  # (batch, num_slots, N_pool)
            selected_indices = selection_probs.argmax(dim=-1)

        # Predict number of exemplars
        num_ex_logits = self.num_exemplars_head(z)  # (batch, num_slots + 1)

        return selection_probs, num_ex_logits, selected_indices


class StructureAwareVAE(nn.Module):
    """Joint VAE for instruction + variable-length exemplar optimization.

    Latent Space (32D total):
        - z_instruction (16D): Instruction content
        - z_exemplar (16D): Exemplar set representation

    Key features:
        - Set Transformer for permutation-invariant exemplar encoding
        - Slot-based decoder for discrete exemplar selection
        - Gumbel-Softmax for differentiable training
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        instruction_latent_dim: int = 16,
        exemplar_latent_dim: int = 16,
        num_slots: int = 8,
        num_inducing: int = 4,
        set_transformer_hidden: int = 128,
        set_transformer_heads: int = 4,
        beta: float = 0.005,
        mse_weight: float = 0.2,
        selection_weight: float = 1.0,
        num_exemplars_weight: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.instruction_latent_dim = instruction_latent_dim
        self.exemplar_latent_dim = exemplar_latent_dim
        self.total_latent_dim = instruction_latent_dim + exemplar_latent_dim
        self.num_slots = num_slots
        self.beta = beta
        self.mse_weight = mse_weight
        self.selection_weight = selection_weight
        self.num_exemplars_weight = num_exemplars_weight

        # Instruction encoder/decoder
        self.instruction_encoder = InstructionEncoder(
            embedding_dim=embedding_dim,
            latent_dim=instruction_latent_dim,
            dropout=dropout,
        )
        self.instruction_decoder = InstructionDecoder(
            latent_dim=instruction_latent_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        # Exemplar encoder/decoder
        self.exemplar_encoder = ExemplarSetEncoder(
            embedding_dim=embedding_dim,
            hidden_dim=set_transformer_hidden,
            latent_dim=exemplar_latent_dim,
            num_inducing=num_inducing,
            num_heads=set_transformer_heads,
            dropout=dropout,
        )
        self.exemplar_decoder = SlotBasedExemplarDecoder(
            latent_dim=exemplar_latent_dim,
            num_slots=num_slots,
            pool_embedding_dim=embedding_dim,
            dropout=dropout,
        )

        # Optional joint refinement layer
        self.joint_refine = nn.Sequential(
            nn.Linear(self.total_latent_dim, self.total_latent_dim),
            nn.GELU(),
            nn.Linear(self.total_latent_dim, self.total_latent_dim),
        )

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Standard VAE reparameterization."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(
        self,
        instruction_emb: torch.Tensor,
        exemplar_embs: torch.Tensor,
        exemplar_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode instruction and exemplars to latent parameters.

        Args:
            instruction_emb: (batch, 768)
            exemplar_embs: (batch, K, 768)
            exemplar_mask: (batch, K), True = valid

        Returns:
            mu_inst, logvar_inst, mu_ex, logvar_ex
        """
        mu_inst, logvar_inst = self.instruction_encoder(instruction_emb)
        mu_ex, logvar_ex = self.exemplar_encoder(exemplar_embs, exemplar_mask)
        return mu_inst, logvar_inst, mu_ex, logvar_ex

    def encode_joint(
        self,
        instruction_emb: torch.Tensor,
        exemplar_embs: torch.Tensor,
        exemplar_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode to joint latent (deterministic mu, for GP).

        Returns:
            z_joint: (batch, total_latent_dim)
        """
        mu_inst, _, mu_ex, _ = self.encode(
            instruction_emb, exemplar_embs, exemplar_mask
        )
        z_joint = torch.cat([mu_inst, mu_ex], dim=-1)

        # Optional refinement
        z_joint = z_joint + self.joint_refine(z_joint)

        return z_joint

    def decode(
        self,
        z_inst: torch.Tensor,
        z_ex: torch.Tensor,
        pool_embeddings: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode latents to instruction embedding and exemplar selection.

        Args:
            z_inst: (batch, instruction_latent_dim)
            z_ex: (batch, exemplar_latent_dim)
            pool_embeddings: (N_pool, 768)
            temperature: Gumbel-Softmax temperature
            hard: Use hard selection

        Returns:
            inst_emb_recon: (batch, 768)
            selection_probs: (batch, num_slots, N_pool)
            num_ex_logits: (batch, num_slots + 1)
            selected_indices: (batch, num_slots)
        """
        inst_emb_recon = self.instruction_decoder(z_inst)
        selection_probs, num_ex_logits, selected_indices = self.exemplar_decoder(
            z_ex, pool_embeddings, temperature=temperature, hard=hard
        )
        return inst_emb_recon, selection_probs, num_ex_logits, selected_indices

    def forward(
        self,
        instruction_emb: torch.Tensor,
        exemplar_embs: torch.Tensor,
        exemplar_mask: Optional[torch.Tensor],
        pool_embeddings: torch.Tensor,
        true_exemplar_indices: torch.Tensor,
        true_num_exemplars: torch.Tensor,
        temperature: float = 1.0,
        beta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Full forward pass with loss computation.

        Args:
            instruction_emb: (batch, 768)
            exemplar_embs: (batch, K, 768)
            exemplar_mask: (batch, K)
            pool_embeddings: (N_pool, 768)
            true_exemplar_indices: (batch, K) ground truth pool indices
            true_num_exemplars: (batch,) ground truth count
            temperature: Gumbel-Softmax temperature
            beta: KL weight (None = use self.beta)

        Returns:
            loss: Total loss scalar
            loss_dict: Individual loss components
        """
        if beta is None:
            beta = self.beta

        # Encode
        mu_inst, logvar_inst, mu_ex, logvar_ex = self.encode(
            instruction_emb, exemplar_embs, exemplar_mask
        )

        # Sample
        z_inst = self.reparameterize(mu_inst, logvar_inst)
        z_ex = self.reparameterize(mu_ex, logvar_ex)

        # Optional joint refinement
        z_joint = torch.cat([z_inst, z_ex], dim=-1)
        z_refined = z_joint + self.joint_refine(z_joint)
        z_inst = z_refined[:, : self.instruction_latent_dim]
        z_ex = z_refined[:, self.instruction_latent_dim :]

        # Decode
        inst_emb_recon, selection_probs, num_ex_logits, _ = self.decode(
            z_inst, z_ex, pool_embeddings, temperature=temperature, hard=False
        )

        # Compute loss
        loss, loss_dict = self.compute_loss(
            instruction_emb=instruction_emb,
            inst_emb_recon=inst_emb_recon,
            mu_inst=mu_inst,
            logvar_inst=logvar_inst,
            mu_ex=mu_ex,
            logvar_ex=logvar_ex,
            selection_probs=selection_probs,
            num_ex_logits=num_ex_logits,
            true_exemplar_indices=true_exemplar_indices,
            true_num_exemplars=true_num_exemplars,
            beta=beta,
        )

        return loss, loss_dict

    def compute_loss(
        self,
        instruction_emb: torch.Tensor,
        inst_emb_recon: torch.Tensor,
        mu_inst: torch.Tensor,
        logvar_inst: torch.Tensor,
        mu_ex: torch.Tensor,
        logvar_ex: torch.Tensor,
        selection_probs: torch.Tensor,
        num_ex_logits: torch.Tensor,
        true_exemplar_indices: torch.Tensor,
        true_num_exemplars: torch.Tensor,
        beta: float,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute VAE loss.

        Total Loss = L_recon + β·L_KL + λ_sel·L_selection + λ_num·L_num_ex

        Components:
            1. Instruction reconstruction (cosine + MSE)
            2. KL divergence for both latent components
            3. Exemplar selection cross-entropy
            4. Number of exemplars cross-entropy
        """
        device = instruction_emb.device
        batch_size = instruction_emb.shape[0]

        # 1. Instruction reconstruction loss
        cosine_sim = F.cosine_similarity(instruction_emb, inst_emb_recon, dim=-1)
        cosine_loss = (1 - cosine_sim).mean()
        mse_loss = F.mse_loss(instruction_emb, inst_emb_recon)
        recon_loss = (1 - self.mse_weight) * cosine_loss + self.mse_weight * mse_loss

        # 2. KL divergence
        kl_inst = -0.5 * (
            1 + logvar_inst - mu_inst.pow(2) - logvar_inst.exp()
        ).sum(dim=-1).mean()
        kl_ex = -0.5 * (
            1 + logvar_ex - mu_ex.pow(2) - logvar_ex.exp()
        ).sum(dim=-1).mean()
        kl_loss = kl_inst + kl_ex

        # 3. Exemplar selection loss
        # selection_probs: (batch, num_slots, N_pool)
        # true_exemplar_indices: (batch, K)
        _, num_slots, N_pool = selection_probs.shape
        K = true_exemplar_indices.shape[1]

        # Compute selection loss only for active slots
        # Use cross-entropy with softmax-ed logits for numerical stability
        selection_loss = torch.tensor(0.0, device=device)
        active_count = 0
        eps = 1e-8

        for b in range(batch_size):
            num_ex = int(true_num_exemplars[b].item())
            for s in range(num_ex):
                if s < K:
                    target_idx = true_exemplar_indices[b, s].long()
                    # Add eps before log for numerical stability
                    slot_log_probs = (selection_probs[b, s] + eps).log()
                    selection_loss = selection_loss - slot_log_probs[target_idx]
                    active_count += 1

        if active_count > 0:
            selection_loss = selection_loss / active_count

        # 4. Number of exemplars loss
        num_ex_loss = F.cross_entropy(num_ex_logits, true_num_exemplars.long())

        # Total loss
        total_loss = (
            recon_loss
            + beta * kl_loss
            + self.selection_weight * selection_loss
            + self.num_exemplars_weight * num_ex_loss
        )

        loss_dict = {
            "total": total_loss.item(),
            "recon": recon_loss.item(),
            "cosine_loss": cosine_loss.item(),
            "mse_loss": mse_loss.item(),
            "kl_total": kl_loss.item(),
            "kl_inst": kl_inst.item(),
            "kl_ex": kl_ex.item(),
            "selection": selection_loss.item(),
            "num_exemplars": num_ex_loss.item(),
            "cosine_mean": cosine_sim.mean().item(),
        }

        return total_loss, loss_dict
