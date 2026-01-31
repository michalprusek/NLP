"""LLM client abstraction - vLLM for local, OpenAI/DeepInfra for API"""
from abc import ABC, abstractmethod
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()


class LLMClient(ABC):
    """Abstract base class for LLM clients"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        pass


def _patch_vllm_platform():
    """Patch vLLM platform detection when NVML is broken but CUDA works."""
    import torch
    if not torch.cuda.is_available():
        return

    try:
        # Force CUDA platform by monkey-patching before vLLM initializes
        import vllm.platforms
        from vllm.platforms.cuda import CudaPlatform

        # Replace the current_platform with CudaPlatform
        cuda_platform = CudaPlatform()
        vllm.platforms.current_platform = cuda_platform
        vllm.platforms._current_platform = cuda_platform
    except (ImportError, AttributeError) as e:
        print(
            f"  Warning: Could not patch vLLM platform ({type(e).__name__}): {e}. "
            "This may cause device detection issues. If you see CUDA errors, "
            "try: pip install --upgrade vllm"
        )


class VLLMClient(LLMClient):
    """LLM client using vLLM for fast local inference"""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
    ):
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["VLLM_TARGET_DEVICE"] = "cuda"

        # Patch vLLM platform before importing LLM
        _patch_vllm_platform()

        from vllm import LLM
        from transformers import AutoTokenizer

        self.model_name = model_name

        print(f"Loading model with vLLM: {model_name}")
        print(f"  Tensor parallel size: {tensor_parallel_size}")
        print(f"  GPU memory utilization: {gpu_memory_utilization:.0%}")

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=4096,
            enforce_eager=tensor_parallel_size > 1, # Disable CUDA graphs
            enable_prefix_caching=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.use_chat_template = (
            hasattr(self.tokenizer, 'chat_template') and
            self.tokenizer.chat_template is not None and
            'Instruct' in model_name
        )
        if self.use_chat_template:
            print("Using chat template (Instruct model)")

    def cleanup(self):
        """
        Properly cleanup vLLM resources and free GPU memory.

        This is critical for switching models on single GPU.
        """
        import gc
        import torch

        if hasattr(self, 'llm') and self.llm is not None:
            # Shutdown the vLLM engine properly
            try:
                # Try to shutdown the engine if method exists
                if hasattr(self.llm, 'llm_engine'):
                    engine = self.llm.llm_engine
                    if hasattr(engine, 'model_executor'):
                        executor = engine.model_executor
                        if hasattr(executor, 'shutdown'):
                            executor.shutdown()
            except KeyboardInterrupt:
                raise  # Never swallow keyboard interrupt
            except RuntimeError as e:
                print(f"  Warning: CUDA runtime error during cleanup (GPU memory may still be allocated): {e}")
            except Exception as e:
                print(f"  Warning: Unexpected error during engine shutdown ({type(e).__name__}): {e}")
                print("  GPU resources may not be fully released - consider restarting if memory issues occur")

            # Delete the LLM object
            del self.llm
            self.llm = None

        # Delete tokenizer
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # NOTE: Do NOT destroy distributed process groups here!
        # vLLM manages its own process groups and destroying them
        # causes "Process group not initialized" errors on next model load.
        # The gc.collect() and empty_cache() below are sufficient.

        # Multiple rounds of garbage collection (important for vLLM)
        for _ in range(3):
            gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()

        print("  vLLM client cleaned up")

    def _format_prompt(self, prompt: str) -> str:
        if not self.use_chat_template:
            return prompt
        messages = [{'role': 'user', 'content': prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, prompt: str, **kwargs) -> str:
        return self.generate_batch([prompt], **kwargs)[0]

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        from vllm import SamplingParams

        max_new_tokens = kwargs.get('max_new_tokens', 512)
        temperature = kwargs.get('temperature', 0.0)
        use_tqdm = kwargs.get('use_tqdm', False)  # Default False to reduce log spam

        formatted = [self._format_prompt(p) for p in prompts]

        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=1.1,
        )

        outputs = self.llm.generate(formatted, params, use_tqdm=use_tqdm)
        return [out.outputs[0].text for out in outputs]


class OpenAIClient(LLMClient):
    """LLM client using OpenAI API (GPT-3.5, GPT-4)"""

    def __init__(self, model_name: str):
        from openai import OpenAI

        self.model_name = model_name
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.client = OpenAI(api_key=api_key)
        print(f"Initialized OpenAI client: {model_name}")

    def generate(self, prompt: str, **kwargs) -> str:
        return self.generate_batch([prompt], **kwargs)[0]

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        max_new_tokens = kwargs.get('max_new_tokens', 512)
        temperature = kwargs.get('temperature', 0.0)

        results = []
        for i, prompt in enumerate(prompts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                if not response.choices:
                    print(f"[ERROR] OpenAI returned empty choices on prompt {i+1}/{len(prompts)} - possible content filter")
                    results.append(None)
                else:
                    results.append(response.choices[0].message.content)
            except KeyboardInterrupt:
                raise  # Never swallow keyboard interrupt
            except Exception as e:
                error_type = type(e).__name__
                print(f"[ERROR] OpenAI {error_type} on prompt {i+1}/{len(prompts)}: {e}")
                # Re-raise authentication errors - these are not recoverable
                if "auth" in error_type.lower() or "401" in str(e):
                    raise
                # For other errors, append None to signal failure (distinguishable from empty response)
                results.append(None)
        return results


class DeepInfraClient(LLMClient):
    """LLM client using DeepInfra API (OpenAI-compatible)"""

    def __init__(self, model_name: str):
        from openai import OpenAI

        self.model_name = model_name
        api_key = os.getenv("DEEPINFRA_API_KEY")
        if not api_key:
            raise ValueError("DEEPINFRA_API_KEY not found in environment")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai"
        )
        print(f"Initialized DeepInfra client: {model_name}")

    def generate(self, prompt: str, **kwargs) -> str:
        return self.generate_batch([prompt], **kwargs)[0]

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        max_new_tokens = kwargs.get('max_new_tokens', 512)
        temperature = kwargs.get('temperature', 0.0)

        results = []
        for i, prompt in enumerate(prompts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                if not response.choices:
                    print(f"[ERROR] DeepInfra returned empty choices on prompt {i+1}/{len(prompts)} - possible content filter")
                    results.append(None)
                else:
                    results.append(response.choices[0].message.content)
            except KeyboardInterrupt:
                raise  # Never swallow keyboard interrupt
            except Exception as e:
                error_type = type(e).__name__
                print(f"[ERROR] DeepInfra {error_type} on prompt {i+1}/{len(prompts)}: {e}")
                # Re-raise authentication errors - these are not recoverable
                if "auth" in error_type.lower() or "401" in str(e):
                    raise
                # For other errors, append None to signal failure (distinguishable from empty response)
                results.append(None)
        return results

class TransformersClient(LLMClient):
    """LLM client using HuggingFace Transformers (Standard PyTorch)"""

    def __init__(self, model_name: str, **kwargs):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model with Transformers: {model_name}")
        print(f"  Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
        )
        self.model.eval()

    def generate(self, prompt: str, **kwargs) -> str:
        return self.generate_batch([prompt], **kwargs)[0]

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        import torch
        
        max_new_tokens = kwargs.get('max_new_tokens', 512)
        temperature = kwargs.get('temperature', 0.0)
        
        # Simple batch generation loop (for robustness, can optimize later if needed)
        results = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )
            
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append(response)
            
        return results
def create_llm_client(model_name: str, backend: str = "auto", **kwargs) -> LLMClient:
    """
    Factory function to create LLM client.

    Args:
        model_name: Model identifier or alias
        backend: 'vllm', 'openai', 'deepinfra', or 'auto'

    Aliases:
        - 'gpt-3.5' -> gpt-3.5-turbo (OpenAI)
        - 'gemma-3-4b' -> google/gemma-3-4b-it (DeepInfra)
    """
    ALIASES = {
        "gpt-3.5": "gpt-3.5-turbo",
        "gemma-3-4b": "google/gemma-3-4b-it",
        "qwen": "Qwen/Qwen2.5-7B-Instruct",
        "llama": "meta-llama/Llama-3.1-8B-Instruct",
    }

    if model_name.lower() in ALIASES:
        resolved = ALIASES[model_name.lower()]
        print(f"Resolved alias '{model_name}' -> '{resolved}'")
        model_name = resolved

    # Auto-detect backend
    if backend == "auto":
        if "gpt" in model_name.lower():
            backend = "openai"
        elif model_name.startswith("google/"):
            backend = "deepinfra"
        else:
            backend = "vllm"

    if backend == "vllm":
        vllm_kwargs = {k: v for k, v in kwargs.items()
                       if k in ['tensor_parallel_size', 'gpu_memory_utilization']}
        return VLLMClient(model_name, **vllm_kwargs)
    elif backend == "transformers":
        return TransformersClient(model_name)
    elif backend == "openai":
        return OpenAIClient(model_name)
    elif backend == "deepinfra":
        return DeepInfraClient(model_name)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use: vllm, transformers, openai, deepinfra")
