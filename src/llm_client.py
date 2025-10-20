"""LLM client abstraction supporting multiple backends"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts"""
        pass


class TransformersClient(LLMClient):
    """LLM client using HuggingFace Transformers"""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        torch_dtype: str = "auto",
    ):
        """
        Initialize Transformers client.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cuda', 'cpu', 'mps', or 'auto')
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            torch_dtype: Data type ('auto', 'float16', 'bfloat16', 'float32')
        """
        import gc

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Only set pad_token if it's not already set
        # IMPORTANT: Do not override existing pad_token (like Qwen2.5 has)!
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Set pad_token = eos_token")
        else:
            print(f"Using model's default pad_token: {self.tokenizer.pad_token}")

        # Use left padding for batch generation (so all prompts end at the same position)
        # This is important for causal LM generation
        if self.tokenizer.padding_side != 'left':
            print(f"Changed padding_side from '{self.tokenizer.padding_side}' to 'left' for generation")
            self.tokenizer.padding_side = 'left'

        # Check if model uses chat template (for Instruct models)
        self.use_chat_template = (
            hasattr(self.tokenizer, 'chat_template') and
            self.tokenizer.chat_template is not None and
            'Instruct' in model_name
        )
        if self.use_chat_template:
            print("Using chat template format (Instruct model detected)")

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        print(f"Using device: {device}")

        # Determine dtype
        if torch_dtype == "auto":
            if device == "mps":
                # bfloat16 is better for MPS on Apple Silicon
                dtype = torch.bfloat16
                print("Using dtype: bfloat16 (recommended for MPS)")
            elif device == "cuda":
                dtype = torch.float16
                print("Using dtype: float16")
            else:
                dtype = torch.float32
                print("Using dtype: float32")
        elif torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float32":
            dtype = torch.float32
        else:
            raise ValueError(f"Unknown torch_dtype: {torch_dtype}")

        # Load model with error handling
        try:
            print("Loading model weights... (this may take a few minutes)")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,  # Changed from torch_dtype to dtype (deprecated)
                device_map=device if device != "mps" else None,
                low_cpu_mem_usage=True,
            )

            if device == "mps":
                print("Moving model to MPS device...")
                self.model = self.model.to(device)

            # Clear memory
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()

            self.model.eval()
            print("Model loaded successfully!")

        except Exception as e:
            print(f"\nError loading model: {e}")
            print("\nTips to reduce memory usage:")
            print("1. Try a smaller model (e.g., Qwen/Qwen2.5-3B-Instruct)")
            print("2. Use CPU instead: --device cpu")
            print("3. Close other applications to free up memory")
            raise

    def _format_prompt(self, prompt: str) -> str:
        """
        Format prompt using chat template if available.

        Args:
            prompt: Raw text prompt

        Returns:
            Formatted prompt (with chat template if applicable)
        """
        if not self.use_chat_template:
            return prompt

        # Use chat template for Instruct models
        messages = [{'role': 'user', 'content': prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a single prompt"""
        return self.generate_batch([prompt], **kwargs)[0]

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of prompts
            **kwargs: Override default generation parameters

        Returns:
            List of generated texts
        """
        # Override defaults with kwargs
        max_new_tokens = kwargs.get('max_new_tokens', self.max_new_tokens)
        temperature = kwargs.get('temperature', self.temperature)
        do_sample = kwargs.get('do_sample', self.do_sample)

        # Fix for very low temperatures causing numerical issues on MPS
        # Use greedy decoding instead of sampling for temp < 0.3
        if temperature < 0.3:
            do_sample = False
            temperature = 1.0  # Temperature is ignored in greedy decoding

        # Format prompts using chat template if needed
        formatted_prompts = [self._format_prompt(p) for p in prompts]

        # DEBUG: Print first formatted prompt to verify chat template
        import os
        if os.environ.get('DEBUG_PROMPTS') == '1' and len(formatted_prompts) > 0:
            print("\n=== DEBUG: First formatted prompt ===")
            print(formatted_prompts[0][:500])
            print("=== END DEBUG ===\n")

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)

        # Generate with numerical stability
        with torch.no_grad():
            gen_kwargs = {
                'max_new_tokens': max_new_tokens,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
            }

            if do_sample:
                gen_kwargs.update({
                    'do_sample': True,
                    'temperature': max(temperature, 0.3),  # Minimum temp 0.3 for stability
                    'top_p': 0.95,
                    'top_k': 50,
                })
            else:
                # Greedy decoding (deterministic)
                gen_kwargs['do_sample'] = False

            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode (skip input tokens)
        # CRITICAL FIX: Use padded input length (same for all items in batch)
        # All inputs are padded to the same length, so we slice at that position
        # skip_special_tokens=True will remove padding tokens from the output
        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[:, input_length:]
        generated_texts = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return generated_texts


class VLLMClient(LLMClient):
    """LLM client using vLLM for fast inference"""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        tensor_parallel_size: int = 1,
    ):
        """
        Initialize vLLM client.

        Args:
            model_name: HuggingFace model identifier
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            tensor_parallel_size: Number of GPUs for tensor parallelism
        """
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print(f"Loading model with vLLM: {model_name}")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype="auto",
        )

        # Load tokenizer to check for chat template support
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Check if model uses chat template (for Instruct models)
        self.use_chat_template = (
            hasattr(self.tokenizer, 'chat_template') and
            self.tokenizer.chat_template is not None and
            'Instruct' in model_name
        )
        if self.use_chat_template:
            print("Using chat template format (Instruct model detected)")

        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
        )

    def _format_prompt(self, prompt: str) -> str:
        """
        Format prompt using chat template if available.

        Args:
            prompt: Raw text prompt

        Returns:
            Formatted prompt (with chat template if applicable)
        """
        if not self.use_chat_template:
            return prompt

        # Use chat template for Instruct models
        messages = [{'role': 'user', 'content': prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a single prompt"""
        return self.generate_batch([prompt], **kwargs)[0]

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of prompts
            **kwargs: Override default generation parameters

        Returns:
            List of generated texts
        """
        from vllm import SamplingParams

        # Override defaults with kwargs
        max_new_tokens = kwargs.get('max_new_tokens', self.max_new_tokens)
        temperature = kwargs.get('temperature', self.temperature)

        # Format prompts using chat template if needed
        formatted_prompts = [self._format_prompt(p) for p in prompts]

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=1.1,  # Penalize repetition
        )

        outputs = self.llm.generate(formatted_prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]


class ClaudeClient(LLMClient):
    """LLM client using Anthropic's Claude API"""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """
        Initialize Claude client.

        Args:
            model_name: Claude model identifier (e.g., 'claude-3-haiku-20240307', 'claude-3-5-sonnet-20241022')
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Environment variables required:
            ANTHROPIC_API_KEY: Your Anthropic API key
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Get API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment variables. "
                "Please add it to your .env file or set it as an environment variable."
            )

        self.client = Anthropic(api_key=api_key)
        print(f"Initialized Claude client with model: {model_name}")

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a single prompt"""
        return self.generate_batch([prompt], **kwargs)[0]

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate text for multiple prompts.

        Args:
            prompts: List of prompts
            **kwargs: Override default generation parameters

        Returns:
            List of generated texts
        """
        # Override defaults with kwargs
        max_new_tokens = kwargs.get('max_new_tokens', self.max_new_tokens)
        temperature = kwargs.get('temperature', self.temperature)

        results = []
        for prompt in prompts:
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                # Extract text from response
                text = response.content[0].text
                results.append(text)
            except Exception as e:
                print(f"Error generating with Claude: {e}")
                results.append("")

        return results


def create_llm_client(
    model_name: str,
    backend: str = "auto",
    **kwargs
) -> LLMClient:
    """
    Factory function to create LLM client.

    Args:
        model_name: Model identifier
        backend: Backend to use ('transformers', 'vllm', 'claude', or 'auto')
                 'auto' will detect Claude models automatically
        **kwargs: Additional arguments for the client

    Returns:
        LLMClient instance
    """
    # Auto-detect Claude models
    if backend == "auto":
        if "claude" in model_name.lower():
            backend = "claude"
        else:
            backend = "transformers"

    if backend == "transformers":
        return TransformersClient(model_name, **kwargs)
    elif backend == "vllm":
        # Filter out parameters not supported by VLLMClient
        vllm_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['max_new_tokens', 'temperature', 'tensor_parallel_size']
        }
        return VLLMClient(model_name, **vllm_kwargs)
    elif backend == "claude":
        # Filter out parameters not supported by ClaudeClient
        claude_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['max_new_tokens', 'temperature']
        }
        return ClaudeClient(model_name, **claude_kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose from: transformers, vllm, claude, auto")
