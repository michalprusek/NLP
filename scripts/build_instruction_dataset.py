#!/usr/bin/env python
"""
Build diverse instruction dataset for flow model training.

This script collects instruction-style prompts from multiple sources:
- General: Alpaca, Dolly, OpenAssistant, FLAN
- Math: OpenMathInstruct, our evaluated prompts
- Code: CodeAlpaca
- Reasoning: Chain-of-thought prompts

Then encodes all with SONAR for flow model training.

Usage:
    # Full dataset (all sources)
    python scripts/build_instruction_dataset.py --output datasets/instruction_embeddings.pt

    # Quick test
    python scripts/build_instruction_dataset.py --output datasets/instruction_embeddings_test.pt \
        --n-alpaca 1000 --n-dolly 1000 --skip-ape
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Optional, Dict

import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_seed_instructions(path: str = "datasets/evaluated_instructions/gsm8k_100_with_embeddings.pt") -> List[str]:
    """Load high-quality seed instructions from warm-start data."""
    data = torch.load(path, weights_only=False)
    instructions = data["instructions"]
    scores = data["accuracies"]

    # Filter to top performers (>80% accuracy)
    good_instructions = [
        inst for inst, score in zip(instructions, scores) if score > 0.80
    ]
    logger.info(f"Loaded {len(good_instructions)} seed instructions with >80% accuracy")
    return good_instructions


def extract_instructions_from_alpaca(n_samples: int = 5000) -> List[str]:
    """Extract instruction-style prompts from Alpaca dataset."""
    from datasets import load_dataset

    logger.info("Loading Alpaca dataset...")
    ds = load_dataset("tatsu-lab/alpaca", split=f"train[:{n_samples}]")

    instructions = []
    for item in ds:
        # Only take instruction field (not input/output)
        inst = item["instruction"].strip()
        if 20 < len(inst) < 500:  # Filter by length
            instructions.append(inst)

    logger.info(f"Extracted {len(instructions)} instructions from Alpaca")
    return instructions


def extract_instructions_from_dolly(n_samples: int = 5000) -> List[str]:
    """Extract instruction-style prompts from Dolly dataset."""
    from datasets import load_dataset

    logger.info("Loading Dolly dataset...")
    ds = load_dataset("databricks/databricks-dolly-15k", split=f"train[:{n_samples}]")

    instructions = []
    for item in ds:
        inst = item["instruction"].strip()
        if 20 < len(inst) < 500:
            instructions.append(inst)

    logger.info(f"Extracted {len(instructions)} instructions from Dolly")
    return instructions


def extract_math_instructions_from_openmath(n_samples: int = 10000) -> List[str]:
    """Extract system-prompt style instructions from OpenMathInstruct solutions."""
    from datasets import load_dataset

    logger.info("Loading OpenMathInstruct-1 dataset...")
    ds = load_dataset("nvidia/OpenMathInstruct-1", split=f"train[:{n_samples}]")

    # Extract instructional patterns from solutions
    instruction_patterns = set()

    for item in ds:
        solution = item["generated_solution"]

        # Extract first sentence if it looks like an instruction
        first_line = solution.split("\n")[0].strip()
        if first_line.startswith("Let's") or first_line.startswith("To solve"):
            instruction_patterns.add(first_line[:200])

    instructions = list(instruction_patterns)
    logger.info(f"Extracted {len(instructions)} instruction patterns from OpenMathInstruct")
    return instructions


def extract_from_openassistant(n_samples: int = 10000) -> List[str]:
    """Extract diverse instructions from OpenAssistant conversations."""
    from datasets import load_dataset

    logger.info("Loading OpenAssistant dataset...")
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
    except Exception:
        ds = load_dataset("timdettmers/openassistant-guanaco", split=f"train[:{n_samples}]")

    instructions = []
    for item in ds:
        # Extract human messages (instructions)
        text = item.get("text", "") or item.get("instruction", "")
        if "### Human:" in text:
            human_parts = text.split("### Human:")
            for part in human_parts[1:]:  # Skip first empty
                inst = part.split("### Assistant:")[0].strip()
                if 20 < len(inst) < 500:
                    instructions.append(inst)
        elif 20 < len(text) < 500:
            instructions.append(text)

        if len(instructions) >= n_samples:
            break

    logger.info(f"Extracted {len(instructions)} instructions from OpenAssistant")
    return instructions[:n_samples]


def extract_from_flan(n_samples: int = 10000) -> List[str]:
    """Extract diverse instructions from FLAN collection."""
    from datasets import load_dataset

    logger.info("Loading FLAN dataset...")
    # FLAN has many subsets, we'll use a few diverse ones
    instructions = []

    try:
        # Natural instructions
        ds = load_dataset("Muennighoff/flan", "niv2", split=f"train[:{n_samples//2}]", trust_remote_code=True)
        for item in ds:
            inst = item.get("inputs", "")
            if 20 < len(inst) < 500:
                instructions.append(inst)
    except Exception as e:
        logger.warning(f"FLAN niv2 failed: {e}")

    try:
        # Chain of thought
        ds = load_dataset("Muennighoff/flan", "cot", split=f"train[:{n_samples//2}]", trust_remote_code=True)
        for item in ds:
            inst = item.get("inputs", "")
            if 20 < len(inst) < 500:
                instructions.append(inst)
    except Exception as e:
        logger.warning(f"FLAN cot failed: {e}")

    logger.info(f"Extracted {len(instructions)} instructions from FLAN")
    return instructions[:n_samples]


def extract_from_wizardlm(n_samples: int = 5000) -> List[str]:
    """Extract complex instructions from WizardLM Evol-Instruct."""
    from datasets import load_dataset

    logger.info("Loading WizardLM dataset...")
    try:
        ds = load_dataset("WizardLM/WizardLM_evol_instruct_70k", split=f"train[:{n_samples}]")
        instructions = []
        for item in ds:
            inst = item.get("instruction", "")
            if 20 < len(inst) < 500:
                instructions.append(inst)
        logger.info(f"Extracted {len(instructions)} instructions from WizardLM")
        return instructions
    except Exception as e:
        logger.warning(f"WizardLM failed: {e}")
        return []


def extract_from_sharegpt(n_samples: int = 5000) -> List[str]:
    """Extract diverse instructions from ShareGPT."""
    from datasets import load_dataset

    logger.info("Loading ShareGPT dataset...")
    try:
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split=f"train[:{n_samples}]")
        instructions = []
        for item in ds:
            convs = item.get("conversations", [])
            for conv in convs:
                if conv.get("from") == "human":
                    inst = conv.get("value", "")
                    if 20 < len(inst) < 500:
                        instructions.append(inst)
                    if len(instructions) >= n_samples:
                        break
            if len(instructions) >= n_samples:
                break
        logger.info(f"Extracted {len(instructions)} instructions from ShareGPT")
        return instructions[:n_samples]
    except Exception as e:
        logger.warning(f"ShareGPT failed: {e}")
        return []


def extract_from_code_alpaca(n_samples: int = 5000) -> List[str]:
    """Extract coding instructions from Code Alpaca."""
    from datasets import load_dataset

    logger.info("Loading Code Alpaca dataset...")
    try:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split=f"train[:{n_samples}]")
        instructions = []
        for item in ds:
            inst = item.get("instruction", "")
            if 20 < len(inst) < 500:
                instructions.append(inst)
        logger.info(f"Extracted {len(instructions)} instructions from Code Alpaca")
        return instructions
    except Exception as e:
        logger.warning(f"Code Alpaca failed: {e}")
        return []


def extract_system_prompts() -> List[str]:
    """Collection of common system prompt patterns."""
    # Hand-crafted diverse system prompts
    prompts = [
        "You are a helpful assistant. Answer the following question step by step.",
        "Think through this problem carefully and show your reasoning.",
        "Break down this problem into smaller steps and solve each one.",
        "Analyze the given information and provide a detailed solution.",
        "Consider all aspects of this question before answering.",
        "Explain your thought process as you work through this problem.",
        "Use logical reasoning to solve the following.",
        "Provide a clear and concise answer with explanation.",
        "Work through this systematically, showing each step.",
        "Think like an expert and solve this problem.",
        "Be thorough in your analysis and provide a complete answer.",
        "Consider edge cases and provide a robust solution.",
        "Solve this step by step, explaining your reasoning at each stage.",
        "Apply critical thinking to answer the following question.",
        "Provide a well-structured response to the following.",
        "Use mathematical reasoning where appropriate.",
        "Think carefully before answering.",
        "Show all your work and explain each step.",
        "Approach this problem methodically.",
        "Give a detailed explanation of your solution.",
    ]
    logger.info(f"Added {len(prompts)} hand-crafted system prompts")
    return prompts


def generate_ape_variations(
    seed_instructions: List[str],
    llm_client,
    n_variations: int = 50,
    temperature: float = 1.0,
) -> List[str]:
    """Generate APE-style variations of seed instructions using LLM."""

    prompt_template = """You are an expert at writing clear instructions for math problem solving.

Here are some highly effective instructions that help AI solve math word problems (each achieved >80% accuracy on GSM8K):

{seed_examples}

Generate {n_variations} NEW and DIVERSE variations of math-solving instructions. Each should:
1. Guide step-by-step mathematical reasoning
2. Be 1-3 sentences long
3. Use different phrasings, structures, and emphases
4. Include variety: some formal, some conversational, some detailed, some brief

Format: Output one instruction per line, numbered 1-{n_variations}.
DO NOT copy the examples verbatim - create original variations."""

    # Sample seed instructions for the prompt
    seed_sample = random.sample(seed_instructions, min(10, len(seed_instructions)))
    seed_text = "\n".join(f"- {inst}" for inst in seed_sample)

    prompt = prompt_template.format(
        seed_examples=seed_text,
        n_variations=n_variations,
    )

    logger.info(f"Generating {n_variations} APE variations...")
    response = llm_client.generate([prompt], temperature=temperature, max_tokens=4000)[0]

    # Parse numbered instructions
    variations = []
    for line in response.split("\n"):
        line = line.strip()
        # Match patterns like "1.", "1)", "1:"
        if line and line[0].isdigit():
            # Remove number prefix
            for sep in [". ", ") ", ": ", "- "]:
                if sep in line[:5]:
                    line = line.split(sep, 1)[-1]
                    break
            if 20 < len(line) < 500:
                variations.append(line)

    logger.info(f"Generated {len(variations)} valid variations")
    return variations


def encode_with_sonar(
    instructions: List[str],
    device: str = "cuda:0",
    batch_size: int = 32,
) -> torch.Tensor:
    """Encode instructions using SONAR encoder."""
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

    device_obj = torch.device(device) if isinstance(device, str) else device
    logger.info(f"Initializing SONAR encoder on {device_obj}...")
    encoder = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=device_obj,
    )

    logger.info(f"Encoding {len(instructions)} instructions...")
    embeddings = encoder.predict(
        instructions,
        source_lang="eng_Latn",
        batch_size=batch_size,
        progress_bar=True,
    )

    logger.info(f"Encoded embeddings shape: {embeddings.shape}")
    return embeddings


def filter_outliers(
    embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    threshold: float = 0.3,
) -> torch.Tensor:
    """Filter embeddings that are too far from reference distribution."""
    # Normalize for cosine similarity
    emb_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
    ref_norm = reference_embeddings / reference_embeddings.norm(dim=1, keepdim=True)

    # Max similarity to any reference point
    sims = emb_norm @ ref_norm.T
    max_sims = sims.max(dim=1).values

    # Keep only embeddings with sufficient similarity
    mask = max_sims > threshold
    filtered = embeddings[mask]

    logger.info(f"Filtered {(~mask).sum().item()} outliers (kept {mask.sum().item()}/{len(embeddings)})")
    return filtered, mask


def main():
    parser = argparse.ArgumentParser(description="Build diverse instruction dataset for flow model")
    parser.add_argument("--output", type=str, default="datasets/instruction_embeddings.pt",
                       help="Output path for embeddings")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Dataset sizes
    parser.add_argument("--n-alpaca", type=int, default=10000, help="Samples from Alpaca")
    parser.add_argument("--n-dolly", type=int, default=10000, help="Samples from Dolly")
    parser.add_argument("--n-openassistant", type=int, default=10000, help="Samples from OpenAssistant")
    parser.add_argument("--n-flan", type=int, default=10000, help="Samples from FLAN")
    parser.add_argument("--n-wizardlm", type=int, default=5000, help="Samples from WizardLM")
    parser.add_argument("--n-sharegpt", type=int, default=5000, help="Samples from ShareGPT")
    parser.add_argument("--n-code-alpaca", type=int, default=5000, help="Samples from Code Alpaca")
    parser.add_argument("--n-openmath", type=int, default=5000, help="Samples from OpenMathInstruct")
    parser.add_argument("--n-ape-variations", type=int, default=500, help="APE variations to generate")

    # Options
    parser.add_argument("--skip-ape", action="store_true", help="Skip APE generation")
    parser.add_argument("--skip-filter", action="store_true", help="Skip outlier filtering")
    parser.add_argument("--filter-threshold", type=float, default=0.15,
                       help="Cosine similarity threshold for filtering (lower = more diverse)")
    args = parser.parse_args()

    sources: Dict[str, int] = {}
    all_instructions = []

    # 1. Load seed instructions (always include)
    seed_instructions = load_seed_instructions()
    all_instructions.extend(seed_instructions)
    sources["seed"] = len(seed_instructions)

    # 2. Hand-crafted system prompts
    system_prompts = extract_system_prompts()
    all_instructions.extend(system_prompts)
    sources["system_prompts"] = len(system_prompts)

    # 3. GENERAL INSTRUCTIONS
    logger.info("\n" + "="*60)
    logger.info("EXTRACTING GENERAL INSTRUCTIONS")
    logger.info("="*60)

    # Alpaca (diverse general tasks)
    if args.n_alpaca > 0:
        alpaca_instructions = extract_instructions_from_alpaca(args.n_alpaca)
        all_instructions.extend(alpaca_instructions)
        sources["alpaca"] = len(alpaca_instructions)

    # Dolly (human-written instructions)
    if args.n_dolly > 0:
        dolly_instructions = extract_instructions_from_dolly(args.n_dolly)
        all_instructions.extend(dolly_instructions)
        sources["dolly"] = len(dolly_instructions)

    # OpenAssistant (conversational)
    if args.n_openassistant > 0:
        oasst_instructions = extract_from_openassistant(args.n_openassistant)
        all_instructions.extend(oasst_instructions)
        sources["openassistant"] = len(oasst_instructions)

    # FLAN (diverse NLP tasks)
    if args.n_flan > 0:
        flan_instructions = extract_from_flan(args.n_flan)
        all_instructions.extend(flan_instructions)
        sources["flan"] = len(flan_instructions)

    # WizardLM (complex evolved instructions)
    if args.n_wizardlm > 0:
        wizard_instructions = extract_from_wizardlm(args.n_wizardlm)
        all_instructions.extend(wizard_instructions)
        sources["wizardlm"] = len(wizard_instructions)

    # ShareGPT (real user queries)
    if args.n_sharegpt > 0:
        sharegpt_instructions = extract_from_sharegpt(args.n_sharegpt)
        all_instructions.extend(sharegpt_instructions)
        sources["sharegpt"] = len(sharegpt_instructions)

    # 4. SPECIALIZED INSTRUCTIONS
    logger.info("\n" + "="*60)
    logger.info("EXTRACTING SPECIALIZED INSTRUCTIONS")
    logger.info("="*60)

    # Code Alpaca (coding tasks)
    if args.n_code_alpaca > 0:
        code_instructions = extract_from_code_alpaca(args.n_code_alpaca)
        all_instructions.extend(code_instructions)
        sources["code_alpaca"] = len(code_instructions)

    # OpenMathInstruct (math)
    if args.n_openmath > 0:
        openmath_instructions = extract_math_instructions_from_openmath(args.n_openmath)
        all_instructions.extend(openmath_instructions)
        sources["openmath"] = len(openmath_instructions)

    # 5. APE-style variations of seed prompts
    if not args.skip_ape and args.n_ape_variations > 0:
        logger.info("\n" + "="*60)
        logger.info("GENERATING APE VARIATIONS")
        logger.info("="*60)
        try:
            from src.llm_client import create_llm_client
            llm = create_llm_client("sonnet", backend="anthropic")

            ape_instructions = []
            for i in range(0, args.n_ape_variations, 50):
                batch_size = min(50, args.n_ape_variations - i)
                variations = generate_ape_variations(seed_instructions, llm, batch_size)
                ape_instructions.extend(variations)

            all_instructions.extend(ape_instructions)
            sources["ape"] = len(ape_instructions)
            logger.info(f"Added {len(ape_instructions)} APE-generated instructions")
        except Exception as e:
            logger.warning(f"APE generation failed: {e}. Continuing without APE variations.")
            sources["ape"] = 0

    # 6. Deduplicate
    logger.info("\n" + "="*60)
    logger.info("DEDUPLICATION AND ENCODING")
    logger.info("="*60)

    unique_instructions = list(set(all_instructions))
    logger.info(f"Total instructions: {len(all_instructions)} → {len(unique_instructions)} unique")

    # 7. Encode with SONAR
    embeddings = encode_with_sonar(unique_instructions, device=args.device)

    # 8. Filter outliers (optional)
    if not args.skip_filter:
        logger.info(f"Filtering outliers (threshold={args.filter_threshold})...")
        # Load warm-start embeddings as reference
        warm_data = torch.load("datasets/evaluated_instructions/gsm8k_100_with_embeddings.pt", weights_only=False)
        reference_embeddings = warm_data["embeddings"]

        embeddings, mask = filter_outliers(embeddings, reference_embeddings, threshold=args.filter_threshold)
        unique_instructions = [inst for inst, m in zip(unique_instructions, mask) if m]

    # 9. Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "embeddings": embeddings,
        "instructions": unique_instructions,
        "sources": sources,
    }, output_path)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Output: {output_path}")
    logger.info(f"Total instructions: {len(unique_instructions)}")
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info("Sources breakdown:")
    for source, count in sources.items():
        logger.info(f"  {source}: {count}")


if __name__ == "__main__":
    main()
