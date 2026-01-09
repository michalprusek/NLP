"""Encoders for BOLT (Simplified).

Components:
- GTREncoder: Sentence-Transformers GTR-T5-Base (768D, L2-normalized)
- InstructionEncoder: 768D → 16D VAE encoder
- InstructionDecoder: 16D → 768D VAE decoder
- ExemplarSetEncoder: Set Transformer for exemplar encoding (from set_transformer.py)
- CrossAttentionScorer: (z_inst, z_ex, pool) → scores → top-8 selection
- StructureAwareVAE: Instruction VAE + Set Transformer + CrossAttentionScorer (fixed K=8)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from sentence_transformers import SentenceTransformer

from bolt.set_transformer import ExemplarSetEncoder


class GTREncoder:
    """GTR-T5-Base encoder for instruction and exemplar embeddings.

    Produces L2-normalized 768D embeddings compatible with Vec2Text.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        try:
            self.model = SentenceTransformer(
                "sentence-transformers/gtr-t5-base",
                device=device,
            )
        except OSError as e:
            if "Connection" in str(e) or "resolve" in str(e) or "404" in str(e):
                raise RuntimeError(
                    f"Failed to download GTR-T5-Base model. Check internet connection.\n"
                    f"If offline, pre-cache with: python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/gtr-t5-base')\"\n"
                    f"Original error: {e}"
                ) from e
            raise
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                raise RuntimeError(
                    f"CUDA error loading GTR-T5-Base on device '{device}': {e}\n"
                    f"Try device='cpu' if CUDA is unavailable."
                ) from e
            raise
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


class CrossAttentionScorer(nn.Module):
    """Score exemplars using cross-attention mechanism.

    z_inst serves as Query (what instruction needs)
    pool_embeddings serve as Key/Value (what exemplars offer)
    z_ex modulates the query (contextual information)

    Advantages over concat+MLP:
    - Natural alignment mechanism between instruction and exemplars
    - Attention weights are interpretable (which exemplars match the query)
    - Multi-head attention captures different aspects of matching
    """

    def __init__(
        self,
        instruction_latent_dim: int = 16,
        exemplar_latent_dim: int = 16,
        pool_embedding_dim: int = 768,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project joint latent to query space
        self.query_proj = nn.Sequential(
            nn.Linear(instruction_latent_dim + exemplar_latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Project pool embeddings to key/value space
        self.key_proj = nn.Linear(pool_embedding_dim, hidden_dim)
        self.value_proj = nn.Linear(pool_embedding_dim, hidden_dim)

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Score projection (value representation → scalar score)
        self.score_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Learnable temperature for attention weight scaling
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        z_inst: torch.Tensor,
        z_ex: torch.Tensor,
        pool_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Score all exemplars using cross-attention.

        Args:
            z_inst: Instruction latent (batch, instruction_latent_dim)
            z_ex: Exemplar latent (batch, exemplar_latent_dim)
            pool_embeddings: All pool exemplars (N_pool, 768)

        Returns:
            scores: (batch, N_pool) - score for each exemplar
        """
        batch_size = z_inst.shape[0]

        # Build query from joint latent
        z_combined = torch.cat([z_inst, z_ex], dim=-1)  # (batch, 32)
        query = self.query_proj(z_combined)  # (batch, hidden_dim)
        query = query.unsqueeze(1)  # (batch, 1, hidden_dim)

        # Build keys and values from pool
        keys = self.key_proj(pool_embeddings)  # (N_pool, hidden_dim)
        values = self.value_proj(pool_embeddings)  # (N_pool, hidden_dim)

        # Expand for batch
        keys = keys.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, N_pool, hidden_dim)
        values = values.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, N_pool, hidden_dim)

        # Cross-attention
        _, attn_weights = self.cross_attn(
            query=query,
            key=keys,
            value=values,
        )
        # attn_weights: (batch, 1, N_pool)

        # Project values through score head
        # Each pool item gets a base score from its representation
        base_scores = self.score_proj(values).squeeze(-1)  # (batch, N_pool)

        # Modulate by attention weights (focus on relevant items)
        # Attention weights indicate query-key alignment
        attn_scores = attn_weights.squeeze(1) * self.temperature  # (batch, N_pool)

        # Combine base scores with attention-based relevance
        scores = base_scores + attn_scores

        return scores

    def select_top_k(
        self,
        z_inst: torch.Tensor,
        z_ex: torch.Tensor,
        pool_embeddings: torch.Tensor,
        k: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select top-k exemplars based on scores.

        Args:
            z_inst: Instruction latent (batch, instruction_latent_dim)
            z_ex: Exemplar latent (batch, exemplar_latent_dim)
            pool_embeddings: All pool exemplars (N_pool, 768)
            k: Number of exemplars to select

        Returns:
            indices: (batch, k) - selected pool indices
            scores: (batch, N_pool) - all scores (for training loss)
        """
        scores = self.forward(z_inst, z_ex, pool_embeddings)
        _, indices = scores.topk(k, dim=-1)
        return indices, scores


class StructureAwareVAE(nn.Module):
    """Simplified VAE for instruction + exemplar optimization.

    Latent Space (32D total):
        - z_instruction (16D): Instruction content
        - z_exemplar (16D): Exemplar set representation

    Architecture features:
        - Set Transformer for permutation-invariant exemplar encoding
        - CrossAttentionScorer for exemplar selection (default)
        - Fixed K=8 exemplars (no variable-length)
        - ListMLE ranking loss for direct rank optimization (default)
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        instruction_latent_dim: int = 16,
        exemplar_latent_dim: int = 16,
        num_exemplars: int = 8,
        num_inducing: int = 4,
        set_transformer_hidden: int = 128,
        set_transformer_heads: int = 4,
        scorer_hidden_dim: int = 128,
        beta: float = 0.005,
        mse_weight: float = 0.2,
        selection_weight: float = 1.0,
        dropout: float = 0.1,
        # Cross-Attention parameters
        cross_attn_heads: int = 4,
        # Ranking loss type
        ranking_loss_type: str = "listmle",  # "listmle" or "bce"
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.instruction_latent_dim = instruction_latent_dim
        self.exemplar_latent_dim = exemplar_latent_dim
        self.total_latent_dim = instruction_latent_dim + exemplar_latent_dim
        self.num_exemplars = num_exemplars
        self.beta = beta
        self.mse_weight = mse_weight
        self.selection_weight = selection_weight
        self.ranking_loss_type = ranking_loss_type

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

        # Exemplar encoder (Set Transformer)
        self.exemplar_encoder = ExemplarSetEncoder(
            embedding_dim=embedding_dim,
            hidden_dim=set_transformer_hidden,
            latent_dim=exemplar_latent_dim,
            num_inducing=num_inducing,
            num_heads=set_transformer_heads,
            dropout=dropout,
        )

        # Exemplar scorer (CrossAttentionScorer is the default)
        self.scorer = CrossAttentionScorer(
            instruction_latent_dim=instruction_latent_dim,
            exemplar_latent_dim=exemplar_latent_dim,
            pool_embedding_dim=embedding_dim,
            hidden_dim=scorer_hidden_dim,
            num_heads=cross_attn_heads,
            dropout=dropout,
        )

        # Joint refinement layer
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

        Args:
            instruction_emb: (batch, 768)
            exemplar_embs: (batch, K, 768)
            exemplar_mask: (batch, K), True = valid

        Returns:
            z_joint: (batch, total_latent_dim)
        """
        mu_inst, _, mu_ex, _ = self.encode(instruction_emb, exemplar_embs, exemplar_mask)
        z_joint = torch.cat([mu_inst, mu_ex], dim=-1)

        # Optional refinement
        z_joint = z_joint + self.joint_refine(z_joint)

        return z_joint

    def decode(
        self,
        z_inst: torch.Tensor,
        z_ex: torch.Tensor,
        pool_embeddings: torch.Tensor,
        k: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode latents to instruction embedding and exemplar selection.

        Args:
            z_inst: (batch, instruction_latent_dim)
            z_ex: (batch, exemplar_latent_dim)
            pool_embeddings: (N_pool, 768)
            k: Number of exemplars to select

        Returns:
            inst_emb_recon: (batch, 768)
            selected_indices: (batch, k)
            scores: (batch, N_pool)
        """
        inst_emb_recon = self.instruction_decoder(z_inst)
        selected_indices, scores = self.scorer.select_top_k(
            z_inst, z_ex, pool_embeddings, k=k
        )
        return inst_emb_recon, selected_indices, scores

    def forward(
        self,
        instruction_emb: torch.Tensor,
        exemplar_embs: torch.Tensor,
        exemplar_mask: torch.Tensor,
        pool_embeddings: torch.Tensor,
        target_exemplar_mask: torch.Tensor,
        beta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Full forward pass with loss computation.

        Args:
            instruction_emb: (batch, 768)
            exemplar_embs: (batch, K, 768) current exemplar embeddings
            exemplar_mask: (batch, K) validity mask
            pool_embeddings: (N_pool, 768)
            target_exemplar_mask: (batch, N_pool) binary mask of good exemplars
            beta: KL weight (None = use self.beta)

        Returns:
            loss: Total loss scalar
            loss_dict: Individual loss components
        """
        if beta is None:
            beta = self.beta

        # Encode (both instruction and exemplars via Set Transformer)
        mu_inst, logvar_inst, mu_ex, logvar_ex = self.encode(
            instruction_emb, exemplar_embs, exemplar_mask
        )

        # Sample
        z_inst = self.reparameterize(mu_inst, logvar_inst)
        z_ex = self.reparameterize(mu_ex, logvar_ex)

        # Joint refinement
        z_joint = torch.cat([z_inst, z_ex], dim=-1)
        z_refined = z_joint + self.joint_refine(z_joint)
        z_inst = z_refined[:, :self.instruction_latent_dim]
        z_ex = z_refined[:, self.instruction_latent_dim:]

        # Decode
        inst_emb_recon = self.instruction_decoder(z_inst)
        scores = self.scorer(z_inst, z_ex, pool_embeddings)

        # Compute loss
        loss, loss_dict = self.compute_loss(
            instruction_emb=instruction_emb,
            inst_emb_recon=inst_emb_recon,
            mu_inst=mu_inst,
            logvar_inst=logvar_inst,
            mu_ex=mu_ex,
            logvar_ex=logvar_ex,
            scores=scores,
            target_exemplar_mask=target_exemplar_mask,
            beta=beta,
        )

        return loss, loss_dict

    def _compute_listmle_loss(
        self,
        scores: torch.Tensor,
        target_exemplar_mask: torch.Tensor,
    ) -> torch.Tensor:
        """ListMLE ranking loss.

        Maximizes likelihood of correct ranking based on target relevance.
        This directly optimizes the ranking order rather than treating each
        exemplar independently like BCE.

        Args:
            scores: (batch, N_pool) predicted scores
            target_exemplar_mask: (batch, N_pool) binary mask (1 = selected exemplar)

        Returns:
            loss: scalar

        Reference: Xia et al., "Listwise Approach to Learning to Rank"
        """
        # Convert binary mask to relevance scores
        relevance = target_exemplar_mask.float()

        # Sort by relevance (descending) to get ideal ranking order
        _, ideal_order = relevance.sort(dim=-1, descending=True)

        # Reorder predicted scores according to ideal ranking
        scores_sorted = scores.gather(1, ideal_order)

        # ListMLE formula:
        # P(π*|s) = Π_{i=1}^{n} exp(s_i) / Σ_{j=i}^{n} exp(s_j)
        # Log likelihood: Σ_{i=1}^{n} [s_i - log(Σ_{j=i}^{n} exp(s_j))]

        # Compute cumulative logsumexp from the end
        # log(Σ_{j=i}^{n} exp(s_j)) for each position i
        scores_reversed = scores_sorted.flip(dims=[1])
        cumsumexp_reversed = torch.logcumsumexp(scores_reversed, dim=1)
        cumsumexp = cumsumexp_reversed.flip(dims=[1])

        # Log likelihood per position
        log_likelihood = scores_sorted - cumsumexp

        # Weight by relevance (focus on correctly ranking selected exemplars)
        # Non-selected items contribute less to the loss
        weights = relevance.gather(1, ideal_order)

        # Weighted sum focuses loss on ranking the K selected exemplars correctly
        weighted_log_likelihood = log_likelihood * weights

        # Negative log likelihood as loss
        loss = -weighted_log_likelihood.sum(dim=1).mean()

        return loss

    def compute_loss(
        self,
        instruction_emb: torch.Tensor,
        inst_emb_recon: torch.Tensor,
        mu_inst: torch.Tensor,
        logvar_inst: torch.Tensor,
        mu_ex: torch.Tensor,
        logvar_ex: torch.Tensor,
        scores: torch.Tensor,
        target_exemplar_mask: torch.Tensor,
        beta: float,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute VAE loss.

        Total Loss = L_recon + β·L_KL + λ_sel·L_selection

        Components:
            1. Instruction reconstruction (cosine + MSE)
            2. KL divergence for both latent components
            3. Exemplar selection loss (ListMLE or BCE based on config)
        """
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

        # 3. Exemplar selection loss (ListMLE or BCE)
        if self.ranking_loss_type == "listmle":
            selection_loss = self._compute_listmle_loss(scores, target_exemplar_mask)
        else:
            # Fallback to BCE for backward compatibility
            selection_loss = F.binary_cross_entropy_with_logits(
                scores,
                target_exemplar_mask.float(),
            )

        # Total loss
        total_loss = (
            recon_loss
            + beta * kl_loss
            + self.selection_weight * selection_loss
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
            "cosine_mean": cosine_sim.mean().item(),
            "ranking_loss_type": self.ranking_loss_type,
        }

        return total_loss, loss_dict
