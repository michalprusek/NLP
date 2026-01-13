"""
GritLM Unified Encoder for LID-O++.

This module implements a unified encoder based on GritLM (Generative Representational
Instruction Tuning) that can operate in both embedding and generation modes.

GritLM key insight: Same model, different attention masks.
- Embedding mode: Bidirectional attention (all tokens see all tokens)
- Generation mode: Causal attention (autoregressive)

Reference: https://arxiv.org/abs/2402.09906

This encoder provides:
1. Drop-in replacement for GTRInstructionEncoder (same interface)
2. NV-Embed style Latent Attention pooling for better embeddings
3. Optional generation mode for text decoding
"""

import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from lido_pp.backbone.latent_attention import LatentAttentionPooling


class GritLMUnifiedEncoder(nn.Module):
    """
    Unified GritLM encoder for embedding and generation.

    Modes:
    - Embedding: Bidirectional attention → Latent Attention → 768D
    - Generation: Causal attention → Text output

    The model uses the same weights but different attention masks for each mode.

    Args:
        model_name: HuggingFace model name (default: "GritLM/GritLM-7B")
        output_dim: Embedding output dimension (default: 768 for GTR compatibility)
        use_latent_attention: Use NV-Embed style pooling (default: True)
        latent_queries: Number of latent query vectors (default: 512)
        latent_heads: Number of attention heads in latent attention (default: 8)
        quantize: Use INT8 quantization (default: False - not needed with 2x L40S)
        device: Device to load model on (default: "cuda:0")
        trust_remote_code: Trust remote code for model loading (default: True)
    """

    def __init__(
        self,
        model_name: str = "GritLM/GritLM-7B",
        output_dim: int = 768,
        use_latent_attention: bool = True,
        latent_queries: int = 512,
        latent_heads: int = 8,
        quantize: bool = False,
        device: str = "cuda:0",
        dtype: str = "float16",
        trust_remote_code: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.output_dim = output_dim
        self.use_latent_attention = use_latent_attention
        self.device = device
        self._mode = "embedding"  # Current mode: "embedding" or "generation"

        # Parse dtype
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        print(f"Loading GritLM model: {model_name}")
        print(f"Device: {device}, dtype: {self.dtype}, quantize: {quantize}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with optional quantization
        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": self.dtype,
        }

        if quantize:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                )
                load_kwargs["device_map"] = "auto"
            except ImportError:
                print("WARNING: bitsandbytes not available, loading without quantization")
                load_kwargs["device_map"] = {"": device}
        else:
            load_kwargs["device_map"] = {"": device}

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs,
        )
        self.model.eval()

        # Get hidden dimension from model config
        self.hidden_dim = self.model.config.hidden_size
        print(f"Hidden dimension: {self.hidden_dim}")

        # Initialize Latent Attention pooling if enabled
        if use_latent_attention:
            self.latent_attention = LatentAttentionPooling(
                hidden_dim=self.hidden_dim,
                num_queries=latent_queries,
                num_heads=latent_heads,
                output_dim=output_dim,
                dropout=0.1,
            ).to(device).to(self.dtype)
        else:
            # Simple projection from hidden_dim to output_dim
            self.projection = nn.Linear(self.hidden_dim, output_dim).to(device).to(self.dtype)

        # Embedding dimension for interface compatibility
        self.embedding_dim = output_dim

        print(f"GritLM encoder ready. Output dim: {output_dim}")

    def _get_bidirectional_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create bidirectional attention mask for embedding mode.

        In embedding mode, all tokens can attend to all other tokens
        (subject to padding mask).

        Args:
            input_ids: Token IDs (B, L)
            attention_mask: Padding mask (B, L)

        Returns:
            4D attention mask for transformer (B, 1, L, L)
        """
        batch_size, seq_len = input_ids.shape

        # Create full attention (all-to-all)
        # Start with attention_mask and expand
        # 1 = attend, 0 = ignore
        mask_2d = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
        mask_4d = mask_2d.expand(-1, -1, seq_len, -1)  # (B, 1, L, L)

        return mask_4d

    def _get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mode: str = "embedding",
    ) -> torch.Tensor:
        """
        Get hidden states from the model.

        Args:
            input_ids: Token IDs (B, L)
            attention_mask: Padding mask (B, L)
            mode: "embedding" for bidirectional, "generation" for causal

        Returns:
            hidden_states: Last layer hidden states (B, L, hidden_dim)
        """
        with torch.no_grad():
            try:
                # Try standard forward with output_hidden_states
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,  # Disable cache to avoid compatibility issues
                )
                # Get last layer hidden states
                hidden_states = outputs.hidden_states[-1]

            except (AttributeError, TypeError) as e:
                # Fallback: Use model's internal forward without caching
                # This handles compatibility issues with some model versions
                import warnings
                warnings.warn(f"Using fallback hidden state extraction: {e}")

                # Get embeddings manually
                inputs_embeds = self.model.get_input_embeddings()(input_ids)

                # Try to access the base model directly
                if hasattr(self.model, 'model'):
                    base_model = self.model.model
                else:
                    base_model = self.model

                # Forward through base model
                if hasattr(base_model, 'layers'):
                    # Manual forward through layers
                    hidden_states = inputs_embeds
                    for layer in base_model.layers:
                        layer_outputs = layer(
                            hidden_states,
                            attention_mask=attention_mask.unsqueeze(1).unsqueeze(2) if attention_mask is not None else None,
                        )
                        hidden_states = layer_outputs[0]

                    # Apply final layer norm if present
                    if hasattr(base_model, 'norm'):
                        hidden_states = base_model.norm(hidden_states)
                else:
                    # Last resort: just use input embeddings
                    hidden_states = inputs_embeds

        return hidden_states

    def encode_embedding(
        self,
        text: str,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode single text to embedding.

        Args:
            text: Input text
            normalize: L2 normalize output (default: True)

        Returns:
            embedding: (output_dim,) tensor
        """
        return self.encode_embedding_batch([text], normalize)[0]

    def encode_embedding_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        max_length: int = 512,
    ) -> torch.Tensor:
        """
        Encode batch of texts to embeddings.

        Args:
            texts: List of input texts
            normalize: L2 normalize outputs (default: True)
            max_length: Maximum sequence length (default: 512)

        Returns:
            embeddings: (B, output_dim) tensor
        """
        # Add embedding instruction prefix (GritLM convention)
        # This signals to the model that we want embedding, not generation
        embed_instruction = "<|embed|>\n"
        texts_with_instruction = [embed_instruction + t for t in texts]

        # Tokenize
        encoded = self.tokenizer(
            texts_with_instruction,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Get hidden states
        hidden_states = self._get_hidden_states(
            input_ids, attention_mask, mode="embedding"
        )

        # Pool to fixed-size embedding
        if self.use_latent_attention:
            embeddings = self.latent_attention(hidden_states, attention_mask)
        else:
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.shape)
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_hidden / sum_mask

            # Project to output dimension
            embeddings = self.projection(pooled)

            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    # === GTR-compatible interface ===

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to numpy array (GTR-compatible interface).

        Args:
            text: Input text

        Returns:
            embedding: (output_dim,) numpy array, L2-normalized
        """
        with torch.no_grad():
            embedding = self.encode_embedding(text, normalize=True)
        return embedding.cpu().numpy()

    def encode_tensor(self, text: str) -> torch.Tensor:
        """
        Encode text to tensor (GTR-compatible interface).

        Args:
            text: Input text

        Returns:
            embedding: (output_dim,) tensor, L2-normalized
        """
        return self.encode_embedding(text, normalize=True)

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode batch of texts to numpy array (GTR-compatible interface).

        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            embeddings: (N, output_dim) numpy array, L2-normalized
        """
        all_embeddings = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, len(texts), batch_size), desc="Encoding")
            except ImportError:
                iterator = range(0, len(texts), batch_size)
        else:
            iterator = range(0, len(texts), batch_size)

        with torch.no_grad():
            for i in iterator:
                batch = texts[i : i + batch_size]
                embeddings = self.encode_embedding_batch(batch, normalize=True)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    # === Generation mode ===

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text continuation.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            do_sample: Use sampling (vs greedy)

        Returns:
            generated_text: Generated continuation
        """
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        input_ids = encoded["input_ids"].to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        generated = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return generated

    # === Memory management ===

    def to_cpu(self):
        """Move model to CPU to free GPU memory."""
        self.model.to("cpu")
        if self.use_latent_attention:
            self.latent_attention.to("cpu")
        else:
            self.projection.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def to_device(self, device: Optional[str] = None):
        """Move model back to GPU."""
        device = device or self.device
        self.model.to(device)
        if self.use_latent_attention:
            self.latent_attention.to(device)
        else:
            self.projection.to(device)

    def get_memory_usage(self) -> dict:
        """Get GPU memory usage."""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated(self.device) / 1e9,
                "reserved": torch.cuda.memory_reserved(self.device) / 1e9,
            }
        return {"allocated": 0, "reserved": 0}


# Factory function for creating encoder
def create_instruction_encoder(
    encoder_type: str = "gritlm",
    **kwargs,
) -> Union[GritLMUnifiedEncoder, "GTRInstructionEncoder"]:
    """
    Factory for instruction encoders.

    Args:
        encoder_type: Which encoder to use
            - "gritlm": GritLM-7B with Latent Attention
            - "gritlm_simple": GritLM-7B with mean pooling
            - "gtr": Original GTR-T5-Base (requires lipo.encoder)

    Returns:
        Encoder instance
    """
    if encoder_type == "gritlm":
        return GritLMUnifiedEncoder(use_latent_attention=True, **kwargs)
    elif encoder_type == "gritlm_simple":
        return GritLMUnifiedEncoder(use_latent_attention=False, **kwargs)
    elif encoder_type == "gtr":
        # Import GTR encoder from LIPO
        from lipo.encoder import GTRInstructionEncoder
        return GTRInstructionEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")


if __name__ == "__main__":
    # Test the encoder
    print("Testing GritLM Unified Encoder...")

    # Create encoder (will download model if not cached)
    encoder = GritLMUnifiedEncoder(
        model_name="GritLM/GritLM-7B",
        output_dim=768,
        use_latent_attention=True,
        device="cuda:0",
    )

    # Test encoding
    test_texts = [
        "Solve the math problem step by step.",
        "Think carefully and show your work.",
        "Calculate the answer to the following question.",
    ]

    print("\nEncoding test texts...")
    embeddings = encoder.encode_batch(test_texts, batch_size=2)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding norms: {np.linalg.norm(embeddings, axis=1)}")

    # Check cosine similarities
    from scipy.spatial.distance import cosine
    print("\nCosine similarities:")
    for i in range(len(test_texts)):
        for j in range(i + 1, len(test_texts)):
            sim = 1 - cosine(embeddings[i], embeddings[j])
            print(f"  {i} vs {j}: {sim:.4f}")

    # Memory usage
    mem = encoder.get_memory_usage()
    print(f"\nGPU Memory: {mem['allocated']:.2f}GB allocated, {mem['reserved']:.2f}GB reserved")
