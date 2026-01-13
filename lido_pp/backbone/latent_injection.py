"""
Latent Injection Decoder for LID-O++.

This module implements "Latent Injection" - a modern, fast approach to converting
latent vectors back to text using GritLM's generative capability.

Key insight: Since GritLM is a unified model (encoder + decoder), we can:
1. Project our 768D latent back to GritLM's hidden space (4096D)
2. Inject this as prefix embedding into inputs_embeds
3. Let GritLM generate tokens autoregressively

This is much faster than iterative optimization approaches (Vec2Text, ZSInvert)
because it uses a single forward pass per token.

Architecture:
    Latent (768D) → Linear Projector → Prefix Embeddings (N × 4096D)
                                      ↓
                            GritLM inputs_embeds
                                      ↓
                            Autoregressive Generation
                                      ↓
                            Reconstructed Text
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class LatentInjectionResult:
    """Result of latent injection decoding."""

    # Generated text
    text: str

    # Generation log probabilities (if available)
    log_probs: Optional[torch.Tensor] = None

    # Number of tokens generated
    num_tokens: int = 0

    # Was generation stopped by EOS or max length?
    stopped_by_eos: bool = False


class LatentProjector(nn.Module):
    """
    Projects latent vectors from embedding space to GritLM hidden space.

    Can project to single vector or sequence of vectors for richer conditioning.

    Architecture options:
    1. Single projection: 768D → 4096D (1 prefix token)
    2. Multi-token projection: 768D → N × 4096D (N prefix tokens)

    Multi-token projection provides more capacity for encoding semantic information.
    """

    def __init__(
        self,
        latent_dim: int = 768,
        hidden_dim: int = 4096,
        num_prefix_tokens: int = 4,
        use_layer_norm: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            latent_dim: Input latent dimension (from encoder)
            hidden_dim: GritLM hidden dimension
            num_prefix_tokens: Number of prefix tokens to generate
            use_layer_norm: Apply layer norm to output
            dropout: Dropout rate
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_prefix_tokens = num_prefix_tokens

        # Total output dimension
        output_dim = hidden_dim * num_prefix_tokens

        # MLP projector: 768D → intermediate → N × 4096D
        intermediate_dim = min(latent_dim * 4, hidden_dim * 2)

        self.projector = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, output_dim),
        )

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = None

        # Learnable scale factor to match token embedding scale
        # Token embeddings typically have std ~0.002, initialized to match
        self.output_scale = nn.Parameter(torch.tensor(10.0))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training with large LLMs."""
        for i, module in enumerate(self.projector):
            if isinstance(module, nn.Linear):
                # Use small initialization to prevent gradient explosion
                # Last layer gets even smaller init since it feeds into the large model
                is_last = (i == len(self.projector) - 1)
                gain = 0.02 if is_last else 0.1
                nn.init.xavier_uniform_(module.weight, gain=gain)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Project latent to prefix embeddings.

        Args:
            latent: Latent vectors (B, latent_dim)

        Returns:
            prefix_embeddings: (B, num_prefix_tokens, hidden_dim)
        """
        batch_size = latent.shape[0]

        # Project
        projected = self.projector(latent)  # (B, num_prefix_tokens * hidden_dim)

        # Reshape to sequence
        prefix = projected.view(batch_size, self.num_prefix_tokens, self.hidden_dim)

        # Apply layer norm if enabled
        if self.layer_norm is not None:
            prefix = self.layer_norm(prefix)

        # Apply learnable scale to match token embedding scale
        prefix = prefix * self.output_scale

        return prefix


class LatentInjectionDecoder(nn.Module):
    """
    Decoder that injects latent vectors into GritLM for text generation.

    This is the "inverse" of the GritLM encoder - it takes latent vectors
    and generates text that should reconstruct the original instruction.

    The decoding process:
    1. Project latent (768D) to prefix embeddings (N × 4096D)
    2. Concatenate with start token embedding
    3. Generate tokens autoregressively

    Key advantages over optimization-based methods:
    - Single forward pass per token (fast)
    - Uses GritLM's language modeling capability
    - Naturally produces fluent, coherent text
    """

    def __init__(
        self,
        gritlm_model: nn.Module,
        tokenizer,
        latent_dim: int = 768,
        num_prefix_tokens: int = 4,
        use_layer_norm: bool = True,
        dropout: float = 0.1,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,  # Match GritLM dtype
    ):
        """
        Args:
            gritlm_model: Pre-loaded GritLM model
            tokenizer: GritLM tokenizer
            latent_dim: Latent dimension from encoder
            num_prefix_tokens: Number of conditioning tokens
            use_layer_norm: Apply layer norm to projector output
            dropout: Dropout rate in projector
            device: Device for projector
            dtype: Data type for projector (should match model)
        """
        super().__init__()

        self.model = gritlm_model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype

        # Get hidden dimension from model
        self.hidden_dim = gritlm_model.config.hidden_size

        # Create projector in float32 for stable gradient computation
        # (output will be cast to model dtype for forward pass)
        self.projector = LatentProjector(
            latent_dim=latent_dim,
            hidden_dim=self.hidden_dim,
            num_prefix_tokens=num_prefix_tokens,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        ).to(device)  # Keep in float32 for mixed precision training

        # Get embedding layer reference for token embeddings
        self.embed_tokens = gritlm_model.get_input_embeddings()

        # Start token for generation (using BOS or a special token)
        if tokenizer.bos_token_id is not None:
            self.start_token_id = tokenizer.bos_token_id
        else:
            # Use instruction prefix as start
            start_tokens = tokenizer.encode("<|user|>", add_special_tokens=False)
            self.start_token_id = start_tokens[0] if start_tokens else tokenizer.eos_token_id

    def _get_start_embedding(self, batch_size: int) -> torch.Tensor:
        """Get embedding for generation start token."""
        start_ids = torch.full(
            (batch_size, 1),
            self.start_token_id,
            dtype=torch.long,
            device=self.device,
        )
        return self.embed_tokens(start_ids)  # (B, 1, hidden_dim)

    def _top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
    ) -> torch.Tensor:
        """Apply top-k and top-p (nucleus) filtering to logits."""
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value

        return logits

    def decode(
        self,
        latent: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        stop_strings: Optional[List[str]] = None,
    ) -> List[LatentInjectionResult]:
        """
        Decode latent vectors to text using manual token-by-token generation.

        This implementation avoids GritLM's model.generate() which has
        compatibility issues with newer transformers DynamicCache.

        Args:
            latent: Latent vectors (B, latent_dim)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            do_sample: Use sampling (vs greedy)
            repetition_penalty: Penalty for repeated tokens
            stop_strings: Strings that stop generation

        Returns:
            List of LatentInjectionResult for each batch element
        """
        batch_size = latent.shape[0]

        # Convert latent to float32 for projector (which is in float32)
        latent_f32 = latent.to(dtype=torch.float32)

        # Project latent to prefix embeddings and cast to model dtype
        prefix_embeds = self.projector(latent_f32)  # (B, num_prefix, hidden_dim), float32
        prefix_embeds = prefix_embeds.to(self.dtype)  # Cast to model dtype for inference

        # Track generated token IDs for each sample
        generated_ids = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        # Current embeddings (starts with prefix)
        current_embeds = prefix_embeds

        with torch.no_grad():
            for step in range(max_new_tokens):
                # Get logits for next token
                outputs = self.model(
                    inputs_embeds=current_embeds,
                    use_cache=False,
                    return_dict=True,
                )

                # Get logits for last position
                next_token_logits = outputs.logits[:, -1, :].float()  # (B, vocab_size)

                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        if generated_ids[i]:
                            for token_id in set(generated_ids[i]):
                                if next_token_logits[i, token_id] < 0:
                                    next_token_logits[i, token_id] *= repetition_penalty
                                else:
                                    next_token_logits[i, token_id] /= repetition_penalty

                # Apply temperature
                if temperature != 1.0 and do_sample:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k/top-p filtering if sampling
                if do_sample:
                    filtered_logits = self._top_k_top_p_filtering(
                        next_token_logits.clone(), top_k=top_k, top_p=top_p
                    )
                    probs = F.softmax(filtered_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # Update generated IDs and check for EOS
                all_finished = True
                for i in range(batch_size):
                    if not finished[i]:
                        token_id = next_tokens[i].item()
                        generated_ids[i].append(token_id)

                        if token_id == self.tokenizer.eos_token_id:
                            finished[i] = True
                        else:
                            all_finished = False
                    else:
                        all_finished = all_finished and True

                if all_finished:
                    break

                # Append new token embeddings for next iteration
                next_token_embeds = self.embed_tokens(next_tokens.unsqueeze(1))  # (B, 1, hidden_dim)
                current_embeds = torch.cat([current_embeds, next_token_embeds], dim=1)

        # Build results
        results = []
        for i in range(batch_size):
            # Decode to text
            token_ids = generated_ids[i]
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

            # Apply stop strings if provided
            if stop_strings:
                for stop in stop_strings:
                    if stop in text:
                        text = text[:text.index(stop)]

            stopped_by_eos = len(token_ids) > 0 and token_ids[-1] == self.tokenizer.eos_token_id

            results.append(LatentInjectionResult(
                text=text,
                num_tokens=len(token_ids),
                stopped_by_eos=stopped_by_eos,
            ))

        return results

    def decode_single(
        self,
        latent: torch.Tensor,
        **kwargs,
    ) -> str:
        """
        Decode single latent to text (convenience method).

        Args:
            latent: Single latent vector (latent_dim,) or (1, latent_dim)
            **kwargs: Passed to decode()

        Returns:
            Generated text string
        """
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        return self.decode(latent, **kwargs)[0].text

    def decode_batch(
        self,
        latents: Union[torch.Tensor, np.ndarray],
        batch_size: int = 8,
        show_progress: bool = True,
        **kwargs,
    ) -> List[str]:
        """
        Decode batch of latents to text.

        Args:
            latents: Latent vectors (N, latent_dim)
            batch_size: Decoding batch size
            show_progress: Show progress bar
            **kwargs: Passed to decode()

        Returns:
            List of generated texts
        """
        if isinstance(latents, np.ndarray):
            latents = torch.tensor(latents, dtype=torch.float32)

        latents = latents.to(self.device)
        n_samples = latents.shape[0]

        all_texts = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, n_samples, batch_size), desc="Decoding")
            except ImportError:
                iterator = range(0, n_samples, batch_size)
        else:
            iterator = range(0, n_samples, batch_size)

        for i in iterator:
            batch = latents[i:i+batch_size]
            results = self.decode(batch, **kwargs)
            all_texts.extend([r.text for r in results])

        return all_texts


class RoundTripEvaluator:
    """
    Evaluates encode-decode round-trip quality.

    Measures how well we can reconstruct text after:
    Text → GritLM Encoder → Latent → Latent Injection → Text'

    Metrics:
    - Semantic similarity (embedding cosine)
    - BLEU/ROUGE scores
    - Exact match rate
    """

    def __init__(
        self,
        encoder,  # GritLMUnifiedEncoder
        decoder: LatentInjectionDecoder,
    ):
        self.encoder = encoder
        self.decoder = decoder

    def evaluate_single(
        self,
        text: str,
        **decode_kwargs,
    ) -> Dict[str, float]:
        """
        Evaluate single text round-trip.

        Args:
            text: Original text
            **decode_kwargs: Passed to decoder

        Returns:
            Dict with quality metrics
        """
        # Encode
        latent = self.encoder.encode_tensor(text)

        # Decode
        reconstructed = self.decoder.decode_single(latent, **decode_kwargs)

        # Re-encode for semantic comparison
        latent_recon = self.encoder.encode_tensor(reconstructed)

        # Cosine similarity
        cosine_sim = F.cosine_similarity(
            latent.unsqueeze(0),
            latent_recon.unsqueeze(0),
        ).item()

        # Exact match
        exact_match = float(text.strip().lower() == reconstructed.strip().lower())

        # Length ratio
        len_ratio = len(reconstructed) / max(len(text), 1)

        return {
            "cosine_similarity": cosine_sim,
            "exact_match": exact_match,
            "length_ratio": len_ratio,
            "original": text,
            "reconstructed": reconstructed,
        }

    def evaluate_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
        **decode_kwargs,
    ) -> Dict[str, float]:
        """
        Evaluate batch of texts.

        Args:
            texts: List of original texts
            batch_size: Processing batch size
            **decode_kwargs: Passed to decoder

        Returns:
            Dict with aggregated metrics
        """
        all_metrics = []

        for text in texts:
            metrics = self.evaluate_single(text, **decode_kwargs)
            all_metrics.append(metrics)

        # Aggregate
        return {
            "mean_cosine_similarity": np.mean([m["cosine_similarity"] for m in all_metrics]),
            "std_cosine_similarity": np.std([m["cosine_similarity"] for m in all_metrics]),
            "exact_match_rate": np.mean([m["exact_match"] for m in all_metrics]),
            "mean_length_ratio": np.mean([m["length_ratio"] for m in all_metrics]),
            "num_samples": len(texts),
        }


class ProjectorTrainer:
    """
    Trainer for the Latent Projector.

    Trains the projector to map latent vectors to prefix embeddings that
    enable good text reconstruction.

    Training objective: Given (text, latent) pairs, minimize cross-entropy
    loss for generating the original text tokens given the projected prefix.

    This is essentially training a "prompt tuning" model where the prompts
    are generated from latent vectors rather than being fixed.
    """

    def __init__(
        self,
        decoder: LatentInjectionDecoder,
        encoder,  # GritLMUnifiedEncoder (for encoding texts during training)
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        max_length: int = 128,
    ):
        """
        Args:
            decoder: LatentInjectionDecoder with projector to train
            encoder: GritLMUnifiedEncoder for encoding texts
            lr: Learning rate
            weight_decay: Weight decay
            max_length: Maximum sequence length for training
        """
        self.decoder = decoder
        self.encoder = encoder
        self.max_length = max_length

        # Only train the projector, freeze GritLM
        for param in decoder.model.parameters():
            param.requires_grad = False

        # Optimizer for projector only
        self.optimizer = torch.optim.AdamW(
            decoder.projector.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Gradient scaler for mixed precision training
        self.scaler = torch.amp.GradScaler('cuda')

        self.device = decoder.device

    def compute_loss(
        self,
        texts: List[str],
        latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute reconstruction loss for batch of texts.

        Args:
            texts: List of target texts to reconstruct
            latents: Optional pre-computed latents (if None, encode texts)

        Returns:
            loss: Cross-entropy reconstruction loss
            metrics: Dict with loss components
        """
        batch_size = len(texts)

        # Encode texts if latents not provided
        if latents is None:
            with torch.no_grad():
                latents = self.encoder.encode_embedding_batch(texts, normalize=True)

        # Ensure latents are on correct device and in float32 for projector
        # (projector is in float32 for stable gradient computation)
        latents = latents.to(device=self.device, dtype=torch.float32)

        # Project to prefix embeddings (float32 projector output)
        prefix_embeds = self.decoder.projector(latents)  # (B, num_prefix, hidden_dim), float32

        # Tokenize target texts
        encoded = self.decoder.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Get token embeddings for targets
        target_embeds = self.decoder.embed_tokens(input_ids)  # (B, L, hidden_dim)

        # Concatenate: [prefix | target_tokens]
        inputs_embeds = torch.cat([prefix_embeds, target_embeds], dim=1)

        # Extend attention mask for prefix
        prefix_mask = torch.ones(
            batch_size, prefix_embeds.shape[1],
            dtype=torch.long, device=self.device
        )
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # Forward through frozen GritLM (without labels - compute loss manually)
        # Note: GritLM's custom forward has a bug where it checks input_ids.shape
        # even when inputs_embeds is provided, so we compute loss manually
        outputs = self.decoder.model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            return_dict=True,
            use_cache=False,  # Disable cache to avoid DynamicCache compatibility issues
        )

        # Get logits and compute cross-entropy loss manually
        # IMPORTANT: Cast to float32 for numerical stability in cross-entropy
        logits = outputs.logits.float()  # (B, prefix_len + seq_len, vocab_size), in float32

        # We want to predict the target tokens from the prefix embeddings
        # Logits at position i predict token at position i+1
        # So logits[:, prefix_len-1:-1] predict input_ids[:, :]

        prefix_len = prefix_embeds.shape[1]
        seq_len = input_ids.shape[1]

        # Get prediction logits (from position prefix_len-1 to end-1)
        pred_logits = logits[:, prefix_len-1:prefix_len-1+seq_len, :]  # (B, seq_len, vocab)

        # Flatten for cross-entropy
        pred_logits_flat = pred_logits.reshape(-1, pred_logits.shape[-1])  # (B*seq_len, vocab)
        target_flat = input_ids.reshape(-1)  # (B*seq_len,)

        # Create mask for non-padding positions
        loss_mask = attention_mask.reshape(-1).float()  # (B*seq_len,)

        # Compute cross-entropy with masking (already in float32)
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(pred_logits_flat, target_flat)  # (B*seq_len,)
        loss = (token_losses * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

        # Compute accuracy
        with torch.no_grad():
            predictions = pred_logits.argmax(dim=-1)  # (B, seq_len)
            correct = (predictions == input_ids).float()
            accuracy = (correct * attention_mask.float()).sum() / attention_mask.sum()

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "perplexity": torch.exp(loss).item(),
        }

        return loss, metrics

    def train_step(
        self,
        texts: List[str],
        latents: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            texts: Batch of texts
            latents: Optional pre-computed latents

        Returns:
            Dict with training metrics
        """
        self.decoder.projector.train()

        self.optimizer.zero_grad()

        # Use mixed precision training with GradScaler for stability
        with torch.amp.autocast('cuda', dtype=torch.float16):
            loss, metrics = self.compute_loss(texts, latents)

        # Scale loss for mixed precision and backward
        self.scaler.scale(loss).backward()

        # Unscale gradients before clipping
        self.scaler.unscale_(self.optimizer)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.decoder.projector.parameters(), 1.0)

        # Update with scaler (handles inf/nan gracefully by skipping update)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return metrics

    def train_epoch(
        self,
        texts: List[str],
        batch_size: int = 8,
        shuffle: bool = True,
    ) -> Dict[str, float]:
        """
        Train for one epoch over texts.

        Args:
            texts: All training texts
            batch_size: Training batch size
            shuffle: Shuffle texts

        Returns:
            Dict with epoch metrics
        """
        import random

        if shuffle:
            texts = texts.copy()
            random.shuffle(texts)

        all_metrics = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            if len(batch) < 2:  # Skip tiny batches
                continue

            metrics = self.train_step(batch)
            all_metrics.append(metrics)

        # Aggregate
        return {
            "epoch_loss": np.mean([m["loss"] for m in all_metrics]),
            "epoch_accuracy": np.mean([m["accuracy"] for m in all_metrics]),
            "epoch_perplexity": np.mean([m["perplexity"] for m in all_metrics]),
            "num_batches": len(all_metrics),
        }


def create_latent_injection_decoder(
    gritlm_encoder,  # GritLMUnifiedEncoder
    num_prefix_tokens: int = 4,
    **kwargs,
) -> LatentInjectionDecoder:
    """
    Factory function to create decoder from encoder.

    Args:
        gritlm_encoder: Initialized GritLMUnifiedEncoder
        num_prefix_tokens: Number of conditioning tokens
        **kwargs: Additional arguments for decoder

    Returns:
        LatentInjectionDecoder instance
    """
    return LatentInjectionDecoder(
        gritlm_model=gritlm_encoder.model,
        tokenizer=gritlm_encoder.tokenizer,
        latent_dim=gritlm_encoder.output_dim,
        num_prefix_tokens=num_prefix_tokens,
        device=gritlm_encoder.device,
        dtype=gritlm_encoder.dtype,  # Match encoder dtype
        **kwargs,
    )


if __name__ == "__main__":
    print("Testing Latent Injection Decoder...")

    # Test projector in isolation
    print("\n1. Testing LatentProjector...")
    projector = LatentProjector(
        latent_dim=768,
        hidden_dim=4096,
        num_prefix_tokens=4,
    )

    latent = torch.randn(2, 768)
    prefix = projector(latent)
    print(f"   Input: {latent.shape}")
    print(f"   Output: {prefix.shape}")
    assert prefix.shape == (2, 4, 4096)

    print(f"   Parameters: {sum(p.numel() for p in projector.parameters()):,}")

    print("\n2. Projector output statistics...")
    print(f"   Mean: {prefix.mean().item():.4f}")
    print(f"   Std: {prefix.std().item():.4f}")

    # Note: Full decoder test requires loaded GritLM model
    print("\n[OK] LatentProjector tests passed!")
    print("\nNote: Full decoder testing requires loaded GritLM model.")
    print("Run with actual model to test full pipeline:")
    print("  from lido_pp.backbone import GritLMUnifiedEncoder")
    print("  from lido_pp.backbone.latent_injection import create_latent_injection_decoder")
    print("  encoder = GritLMUnifiedEncoder()")
    print("  decoder = create_latent_injection_decoder(encoder)")
    print("  text = decoder.decode_single(encoder.encode_tensor('Test instruction'))")
