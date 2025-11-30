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


class VLLMClient(LLMClient):
    """LLM client using vLLM for fast local inference"""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
    ):
        from vllm import LLM
        from transformers import AutoTokenizer

        self.model_name = model_name

        print(f"Loading model with vLLM: {model_name}")
        print(f"  Tensor parallel size: {tensor_parallel_size}")
        print(f"  GPU memory utilization: {gpu_memory_utilization:.0%}")

        os.environ["VLLM_USE_V1"] = "0"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

        formatted = [self._format_prompt(p) for p in prompts]

        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=1.1,
        )

        outputs = self.llm.generate(formatted, params, use_tqdm=True)
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
        for prompt in prompts:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                results.append(response.choices[0].message.content)
            except Exception as e:
                print(f"OpenAI error: {e}")
                results.append("")
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
        for prompt in prompts:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                results.append(response.choices[0].message.content)
            except Exception as e:
                print(f"DeepInfra error: {e}")
                results.append("")
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
    elif backend == "openai":
        return OpenAIClient(model_name)
    elif backend == "deepinfra":
        return DeepInfraClient(model_name)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use: vllm, openai, deepinfra")
