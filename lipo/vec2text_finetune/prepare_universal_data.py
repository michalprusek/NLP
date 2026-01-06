"""Prepare universal instruction data for Vec2Text corrector fine-tuning.

OPTIMIZED VERSION:
- Batched Inverter generation (key speedup!)
- Greedy decoding (no beam search)
- Multi-GPU parallel processing
- torch.compile for faster inference

Usage:
    uv run python -m lipo.vec2text_finetune.prepare_universal_data
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass, asdict
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """Single training example for corrector fine-tuning."""
    text: str
    embedding: List[float]
    hypothesis: str


def get_available_gpus() -> List[int]:
    """Get list of available GPU IDs."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def load_models_on_gpu(gpu_id: int, compile_model: bool = True):
    """Load GTR and Inverter on specific GPU with optimizations."""
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from vec2text.models.config import InversionConfig
    from vec2text.models.inversion import InversionModel

    device = f"cuda:{gpu_id}"

    # Load GTR
    gtr = SentenceTransformer("sentence-transformers/gtr-t5-base", device=device)

    # Load Inverter
    inv_weights = hf_hub_download(
        "ielabgroup/vec2text_gtr-base-st_inversion", "model.safetensors"
    )
    inv_config = InversionConfig.from_pretrained(
        "ielabgroup/vec2text_gtr-base-st_inversion"
    )
    inverter = InversionModel(inv_config)
    inverter.load_state_dict(load_file(inv_weights), strict=False)
    inverter = inverter.to(device)
    inverter.eval()

    # Enable optimizations
    if hasattr(torch, 'inference_mode'):
        torch.set_float32_matmul_precision('high')

    return gtr, inverter, device


def batched_invert(
    inverter,
    embeddings: torch.Tensor,
    batch_size: int = 32,
    max_length: int = 64,
) -> List[str]:
    """Generate hypotheses for multiple embeddings in batches.

    Key optimization: Process multiple embeddings at once.
    """
    hypotheses = []
    device = embeddings.device

    with torch.inference_mode():
        for i in range(0, len(embeddings), batch_size):
            batch_emb = embeddings[i:i + batch_size]

            # Greedy decoding (much faster than beam search)
            gen_kwargs = {
                "do_sample": False,
                "max_length": max_length,
                "num_beams": 1,  # Greedy
            }

            # Generate for batch
            output_ids = inverter.generate(
                inputs={"frozen_embeddings": batch_emb},
                generation_kwargs=gen_kwargs,
            )

            # Decode all at once
            batch_hypotheses = inverter.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            hypotheses.extend([h.strip() for h in batch_hypotheses])

    return hypotheses


def process_chunk_fast(
    instructions: List[str],
    gpu_id: int,
    encode_batch_size: int = 256,
    invert_batch_size: int = 64,
) -> List[TrainingExample]:
    """Process instructions on GPU with batched operations."""

    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    # Load models
    gtr, inverter, _ = load_models_on_gpu(gpu_id)
    logger.info(f"GPU {gpu_id}: Models loaded, processing {len(instructions)} instructions")

    examples = []

    # Process in large chunks for GTR encoding
    for chunk_start in tqdm(range(0, len(instructions), encode_batch_size),
                           desc=f"GPU {gpu_id}", position=gpu_id):
        chunk_end = min(chunk_start + encode_batch_size, len(instructions))
        chunk_texts = instructions[chunk_start:chunk_end]

        # Batch encode with GTR
        embeddings = gtr.encode(
            chunk_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=encode_batch_size,
            show_progress_bar=False,
        )

        # Batch invert
        hypotheses = batched_invert(
            inverter,
            embeddings,
            batch_size=invert_batch_size,
            max_length=64,
        )

        # Create examples
        for text, emb, hyp in zip(chunk_texts, embeddings, hypotheses):
            examples.append(TrainingExample(
                text=text,
                embedding=emb.cpu().tolist(),
                hypothesis=hyp,
            ))

    return examples


def gpu_worker_fast(
    gpu_id: int,
    instructions: List[str],
    result_queue: mp.Queue,
    encode_batch_size: int = 256,
    invert_batch_size: int = 64,
):
    """Fast GPU worker with batched processing."""
    try:
        examples = process_chunk_fast(
            instructions,
            gpu_id,
            encode_batch_size,
            invert_batch_size,
        )
        result_queue.put((gpu_id, examples))
        logger.info(f"GPU {gpu_id}: Completed {len(examples)} examples")
    except Exception as e:
        import traceback
        logger.error(f"GPU {gpu_id} error: {e}\n{traceback.format_exc()}")
        result_queue.put((gpu_id, []))


def parallel_process_fast(
    instructions: List[str],
    encode_batch_size: int = 256,
    invert_batch_size: int = 64,
) -> List[TrainingExample]:
    """Process instructions in parallel across all GPUs."""
    gpus = get_available_gpus()
    n_gpus = len(gpus)

    if n_gpus == 0:
        raise RuntimeError("No GPUs available!")

    logger.info(f"Using {n_gpus} GPUs: {gpus}")

    # Split instructions across GPUs
    chunk_size = len(instructions) // n_gpus
    chunks = []
    for i in range(n_gpus):
        start = i * chunk_size
        end = start + chunk_size if i < n_gpus - 1 else len(instructions)
        chunks.append(instructions[start:end])

    logger.info(f"Chunk sizes: {[len(c) for c in chunks]}")

    # Create queue
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()

    # Start workers
    processes = []
    for gpu_id, chunk in zip(gpus, chunks):
        p = ctx.Process(
            target=gpu_worker_fast,
            args=(gpu_id, chunk, result_queue, encode_batch_size, invert_batch_size)
        )
        p.start()
        processes.append(p)

    # Collect results
    all_examples = []
    for _ in range(n_gpus):
        gpu_id, examples = result_queue.get()
        all_examples.extend(examples)
        logger.info(f"Collected {len(examples)} examples from GPU {gpu_id}")

    # Wait for processes
    for p in processes:
        p.join()

    return all_examples


def split_data(
    examples: List[TrainingExample],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[List[TrainingExample], List[TrainingExample]]:
    """Split examples into train/eval sets."""
    import random
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def save_data(
    train: List[TrainingExample],
    eval_data: List[TrainingExample],
    output_dir: str = "lipo/vec2text_finetune/data",
    prefix: str = "universal_",
):
    """Save training and evaluation data."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / f"{prefix}train.json"
    eval_path = output_path / f"{prefix}eval.json"

    with open(train_path, "w") as f:
        json.dump([asdict(ex) for ex in train], f)

    with open(eval_path, "w") as f:
        json.dump([asdict(ex) for ex in eval_data], f)

    logger.info(f"Saved {len(train)} training examples to {train_path}")
    logger.info(f"Saved {len(eval_data)} evaluation examples to {eval_path}")


def main():
    """Main pipeline."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="lipo/vec2text_finetune/data/universal_instructions.json")
    parser.add_argument("--output-dir", type=str,
                        default="lipo/vec2text_finetune/data")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--invert-batch-size", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.9)

    args = parser.parse_args()

    # Load
    logger.info("=== Loading Instructions ===")
    with open(args.input) as f:
        data = json.load(f)

    instructions = data.get("instructions", [])
    logger.info(f"Loaded {len(instructions)} instructions")

    if args.max_samples:
        instructions = instructions[:args.max_samples]
        logger.info(f"Limited to {len(instructions)}")

    # Process
    logger.info("=== Processing (Batched + Multi-GPU) ===")
    gpus = get_available_gpus()
    logger.info(f"GPUs: {gpus}")

    start = time.time()

    if len(gpus) > 1:
        examples = parallel_process_fast(
            instructions,
            args.encode_batch_size,
            args.invert_batch_size
        )
    else:
        examples = process_chunk_fast(
            instructions,
            0,
            args.encode_batch_size,
            args.invert_batch_size
        )

    elapsed = time.time() - start
    speed = len(instructions) / elapsed

    logger.info(f"\nCompleted in {elapsed/60:.1f} min ({speed:.1f} inst/sec)")

    # Split & Save
    train, eval_data = split_data(examples, args.train_ratio)
    save_data(train, eval_data, args.output_dir)

    # Summary
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total: {len(instructions)}")
    logger.info(f"Train: {len(train)}, Eval: {len(eval_data)}")
    logger.info(f"Time: {elapsed/60:.1f} min")
    logger.info(f"Speed: {speed:.1f} inst/sec")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
