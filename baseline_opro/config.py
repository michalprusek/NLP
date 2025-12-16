"""
Baseline OPRO configuration - same models and budget as multi_model_optimizer
"""

# Models (same as multi_model_optimizer)
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-3-4b-it",
]

# Budget configuration
TOTAL_BUDGET = 200_000  # Same as multi_model_optimizer
PER_MODEL_BUDGET = TOTAL_BUDGET // len(MODELS)  # 50,000 per model

# Evaluation settings
MINIBATCH_SIZE = 261  # Fixed eval set size (~3.5% of GSM8K train)
NUM_CANDIDATES_PER_ITER = 8  # Default OPRO setting

# Hardware
GPU_ID = "1"  # Second GPU
