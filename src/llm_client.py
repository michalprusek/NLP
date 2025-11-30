"""LLM client abstraction supporting multiple backends"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import time
import signal
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TimeoutError(Exception):
    """Raised when an operation times out"""
    pass


@contextmanager
def timeout_context(seconds: int, error_message: str = "Operation timed out"):
    """Context manager for timing out operations using SIGALRM (Unix only)"""
    def timeout_handler(signum, frame):
        raise TimeoutError(error_message)

    # Store old handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


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
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
    ):
        """
        Initialize vLLM client.

        Args:
            model_name: HuggingFace model identifier
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        """
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print(f"Loading model with vLLM: {model_name}")
        print(f"  Tensor parallel size: {tensor_parallel_size}")
        print(f"  GPU memory utilization: {gpu_memory_utilization:.0%}")

        # Disable v1 engine for compatibility (use stable v0 engine)
        # v1 engine is experimental and has issues with some models
        import os
        os.environ["VLLM_USE_V1"] = "0"

        # Set PyTorch CUDA allocator to avoid fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Disable CUDA graphs when using tensor parallelism to avoid custom_all_reduce errors
        # CUDA graphs can cause issues with multi-GPU tensor parallelism
        enforce_eager = tensor_parallel_size > 1

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=4096,  # Limit context length to reduce KV cache size
            enforce_eager=enforce_eager,  # Use eager mode for tensor parallelism
            enable_prefix_caching=True,  # CRITICAL: Cache KV for shared prefixes (instructions/exemplars)
        )

        # Warmup run to compile CUDA graphs and allocate KV cache
        self._warmup_done = False

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

    def warmup(self, sample_prompts: list[str] = None) -> None:
        """
        Explicitly warm up the model by running a few inference passes.

        This compiles CUDA graphs and allocates KV cache, eliminating the
        "slow first 200 samples" problem. Call this ONCE after initialization
        with representative prompts (same length as actual prompts).

        Args:
            sample_prompts: Optional list of sample prompts to warm up with.
                           If None, uses a default warmup prompt.
        """
        if self._warmup_done:
            return

        print("Warming up vLLM engine (compiling CUDA graphs)...")

        if sample_prompts is None:
            # Default warmup with varying sequence lengths to cover common cases
            sample_prompts = [
                "Solve: 2 + 2 = ?" * 10,  # Short
                "Solve step by step: If Alice has 5 apples and gives 2 to Bob, how many does she have? " * 5,  # Medium
                "Detailed math problem: " + "x " * 500,  # Long (to warm up longer contexts)
            ]

        from vllm import SamplingParams
        warmup_params = SamplingParams(max_tokens=10, temperature=0.0)

        # Format prompts
        formatted = [self._format_prompt(p) for p in sample_prompts]

        # Run warmup - this compiles CUDA graphs
        import time
        start = time.time()
        _ = self.llm.generate(formatted, warmup_params, use_tqdm=False)
        elapsed = time.time() - start

        print(f"Warmup complete in {elapsed:.2f}s - subsequent inference will be faster")
        self._warmup_done = True

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
        # Solve this math problem step by step.
        # "<|im_start|>user\nSolve this math problem step by step.\n\nQuestion: 2+2\nAnswer:<|im_end|>\n<|im_start|>assistant\n"
        messages = [{'role': 'user', 'content': prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a single prompt"""
        return self.generate_batch([prompt], **kwargs)[0]

    def generate_batch(
        self,
        prompts: List[str],
        timeout_seconds: int = 300,
        max_retries: int = 2,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts with timeout and retry.

        Args:
            prompts: List of prompts
            timeout_seconds: Timeout per batch in seconds (default: 5 min)
            max_retries: Number of retries on timeout (default: 2)
            **kwargs: Override default generation parameters

        Returns:
            List of generated texts
        """
        from vllm import SamplingParams

        # Auto-warmup on first call if not done yet
        if not self._warmup_done:
            self.warmup(prompts[:3] if len(prompts) >= 3 else prompts)

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

        # Retry loop with timeout
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Use timeout context to prevent infinite hangs
                with timeout_context(
                    timeout_seconds,
                    f"vLLM generate timed out after {timeout_seconds}s (batch size: {len(prompts)})"
                ):
                    outputs = self.llm.generate(formatted_prompts, sampling_params, use_tqdm=True)
                    return [output.outputs[0].text for output in outputs]
            except TimeoutError as e:
                last_error = e
                if attempt < max_retries:
                    print(f"Warning: vLLM timeout on attempt {attempt + 1}/{max_retries + 1}, retrying...")
                    # Small delay before retry
                    time.sleep(2)
                else:
                    print(f"Error: vLLM timed out after {max_retries + 1} attempts")
                    raise
            except Exception as e:
                # Non-timeout errors are not retried
                raise

        # Should not reach here, but just in case
        raise last_error if last_error else RuntimeError("Unknown error in generate_batch")


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


class OpenAIClient(LLMClient):
    """LLM client using OpenAI's API"""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ):
        """
        Initialize OpenAI client.

        Args:
            model_name: OpenAI model identifier (e.g., 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo')
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Environment variables required:
            OPENAI_API_KEY: Your OpenAI API key
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please add it to your .env file or set it as an environment variable."
            )

        self.client = OpenAI(api_key=api_key)
        print(f"Initialized OpenAI client with model: {model_name}")

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
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                # Extract text from response
                text = response.choices[0].message.content
                results.append(text)
            except Exception as e:
                print(f"Error generating with OpenAI: {e}")
                results.append("")

        return results


class DeepInfraClient(LLMClient):
    """LLM client using DeepInfra's OpenAI-compatible API"""

    # DeepInfra model aliases
    MODEL_ALIASES = {
        "gemma-3-4b": "google/gemma-3-4b-it",
        "gemma-3-27b": "google/gemma-3-27b-it",
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
        "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    }

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ):
        """
        Initialize DeepInfra client.

        Args:
            model_name: DeepInfra model identifier (e.g., 'google/gemma-3-4b-it')
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Environment variables required:
            DEEPINFRA_API_KEY: Your DeepInfra API key
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        # Resolve alias if present
        if model_name.lower() in self.MODEL_ALIASES:
            resolved = self.MODEL_ALIASES[model_name.lower()]
            print(f"Resolved DeepInfra alias '{model_name}' -> '{resolved}'")
            model_name = resolved

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Get API key from environment
        api_key = os.getenv("DEEPINFRA_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPINFRA_API_KEY not found in environment variables. "
                "Please add it to your .env file or set it as an environment variable."
            )

        # Use OpenAI client with DeepInfra base URL
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai"
        )
        print(f"Initialized DeepInfra client with model: {model_name}")

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
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                # Extract text from response
                text = response.choices[0].message.content
                results.append(text)
            except Exception as e:
                print(f"Error generating with DeepInfra: {e}")
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
        model_name: Model identifier. Supports aliases:
                    - 'haiku' -> Claude Haiku 4.5
                    - 'sonnet' -> Claude Sonnet 4.5
                    - 'gemma-3-4b' -> google/gemma-3-4b-it (DeepInfra)
                    - 'gemma-3-27b' -> google/gemma-3-27b-it (DeepInfra)
        backend: Backend to use ('transformers', 'vllm', 'claude', 'openai', 'deepinfra', or 'auto')
                 'auto' will detect Claude, OpenAI, and DeepInfra models automatically
        **kwargs: Additional arguments for the client

    Returns:
        LLMClient instance
    """
    # Model aliases for Claude and OpenAI models (always use latest versions)
    MODEL_ALIASES = {
        "haiku": "claude-haiku-4-5-20251001",      # Latest Haiku (4.5)
        "sonnet": "claude-sonnet-4-5-20251022",    # Latest Sonnet (4.5)
        "gpt-3.5": "gpt-3.5-turbo",                # OpenAI GPT-3.5
        "gpt-4": "gpt-4-turbo",                    # OpenAI GPT-4 Turbo
        # DeepInfra models
        "gemma-3-4b": "google/gemma-3-4b-it",      # Gemma 3 4B via DeepInfra
        "gemma-3-27b": "google/gemma-3-27b-it",    # Gemma 3 27B via DeepInfra
    }

    # Resolve alias if present
    original_model_name = model_name
    if model_name.lower() in MODEL_ALIASES:
        model_name = MODEL_ALIASES[model_name.lower()]
        print(f"Resolved model alias '{original_model_name}' -> '{model_name}'")

    # Auto-detect model backend
    if backend == "auto":
        if "claude" in model_name.lower():
            backend = "claude"
        elif "gpt" in model_name.lower():
            backend = "openai"
        elif model_name.startswith("google/") or model_name.startswith("meta-llama/Meta"):
            # DeepInfra hosted models (Gemma, Llama via API)
            backend = "deepinfra"
        else:
            backend = "transformers"

    if backend == "transformers":
        return TransformersClient(model_name, **kwargs)
    elif backend == "vllm":
        # Filter out parameters not supported by VLLMClient
        vllm_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['max_new_tokens', 'temperature', 'tensor_parallel_size', 'gpu_memory_utilization']
        }
        return VLLMClient(model_name, **vllm_kwargs)
    elif backend == "claude":
        # Filter out parameters not supported by ClaudeClient
        claude_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['max_new_tokens', 'temperature']
        }
        return ClaudeClient(model_name, **claude_kwargs)
    elif backend == "openai":
        # Filter out parameters not supported by OpenAIClient
        openai_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['max_new_tokens', 'temperature']
        }
        return OpenAIClient(model_name, **openai_kwargs)
    elif backend == "deepinfra":
        # Filter out parameters not supported by DeepInfraClient
        deepinfra_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ['max_new_tokens', 'temperature']
        }
        return DeepInfraClient(model_name, **deepinfra_kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose from: transformers, vllm, claude, openai, deepinfra, auto")
