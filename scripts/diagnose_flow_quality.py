#!/usr/bin/env python3
"""
Diagnose flow quality: why flow-generated prompts achieve high accuracy but decode incoherently.

Performs three key analyses:
1. Distribution Comparison: Flow samples vs instruction embeddings
2. Round-Trip Fidelity: L2-r metric for on/off-manifold detection
3. SONAR Decoder Baseline: Verify decoder works on known-good embeddings

Outputs JSON report with actionable metrics and root cause diagnosis.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose flow quality and identify root cause of incoherent decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--flow-checkpoint",
        type=str,
        default=None,
        help="Flow checkpoint path. If None, auto-detects latest.",
    )
    parser.add_argument(
        "--instruction-data",
        type=str,
        default="datasets/gsm8k_instructions_vs.pt",
        help="Path to instruction embeddings dataset",
    )
    parser.add_argument(
        "--bo-checkpoint",
        type=str,
        default=None,
        help="BO checkpoint for high-accuracy samples analysis",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of flow samples to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/diagnostics/flow_quality_report.json",
        help="Output path for JSON diagnostic report",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for computation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def find_flow_checkpoint() -> str:
    """Auto-detect latest flow checkpoint."""
    results_dir = Path("results")
    flow_dirs = sorted(results_dir.glob("flow_ot_*"))
    if not flow_dirs:
        raise FileNotFoundError("No flow_ot_* directories found in results/")

    latest_dir = flow_dirs[-1]
    checkpoint = latest_dir / "checkpoint_final.pt"
    if checkpoint.exists():
        return str(checkpoint)

    # Fall back to latest epoch checkpoint
    checkpoints = sorted(latest_dir.glob("checkpoint_epoch_*.pt"))
    if checkpoints:
        return str(checkpoints[-1])

    raise FileNotFoundError(f"No checkpoints found in {latest_dir}")


def load_flow_model(checkpoint_path: str, device: str):
    """Load flow model from checkpoint."""
    import torch
    from src.ecoflow.flow_model import FlowMatchingModel
    from src.ecoflow.velocity_network import VelocityNetwork

    logging.info(f"Loading flow checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract config
    config = checkpoint.get("config", {})
    input_dim = config.get("input_dim", 1024)
    hidden_dim = config.get("hidden_dim", 512)
    num_layers = config.get("num_layers", 6)

    # Build model
    velocity_net = VelocityNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    velocity_net.load_state_dict(checkpoint["model_state_dict"])
    velocity_net.to(device)
    velocity_net.eval()

    # Get normalization stats
    norm_stats = checkpoint.get("norm_stats", None)
    if norm_stats is not None:
        norm_stats = {k: v.to(device) for k, v in norm_stats.items()}

    model = FlowMatchingModel(velocity_net, norm_stats)
    logging.info(f"Loaded flow model: {input_dim}D, {hidden_dim} hidden, {num_layers} layers")

    return model


def compute_distribution_metrics(flow_samples, instruction_embeddings):
    """
    Compute distribution comparison metrics between flow samples and instructions.

    Returns dict with:
    - mean_mse: Average MSE to instruction centroid
    - norm_diff: Difference in L2 norm distributions
    - cosine_to_centroid: Cosine similarity to instruction centroid
    - wasserstein_1d: 1D Wasserstein distance (projected)
    - cluster_overlap: Fraction of flow samples in instruction clusters
    """
    import torch
    from scipy import stats
    from sklearn.cluster import KMeans
    import numpy as np

    # Ensure tensors
    flow = flow_samples if isinstance(flow_samples, torch.Tensor) else torch.tensor(flow_samples)
    instr = instruction_embeddings if isinstance(instruction_embeddings, torch.Tensor) else torch.tensor(instruction_embeddings)

    flow_np = flow.cpu().numpy()
    instr_np = instr.cpu().numpy()

    # Centroid of instruction embeddings
    instr_centroid = instr.mean(dim=0)

    # Mean MSE to instruction centroid
    mse_to_centroid = ((flow - instr_centroid) ** 2).mean(dim=1)
    mean_mse = mse_to_centroid.mean().item()

    # L2 norm distributions
    flow_norms = flow.norm(dim=1).cpu().numpy()
    instr_norms = instr.norm(dim=1).cpu().numpy()
    norm_diff = abs(flow_norms.mean() - instr_norms.mean())
    norm_diff_std = abs(flow_norms.std() - instr_norms.std())

    # Cosine similarity to centroid
    centroid_normalized = instr_centroid / instr_centroid.norm()
    flow_normalized = flow / flow.norm(dim=1, keepdim=True)
    cosine_to_centroid = (flow_normalized @ centroid_normalized).mean().item()

    # 1D Wasserstein distance (project onto principal axis)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(instr_np)
    flow_proj = pca.transform(flow_np).flatten()
    instr_proj = pca.transform(instr_np).flatten()
    wasserstein_1d = stats.wasserstein_distance(flow_proj, instr_proj)

    # K-means clustering: check if flow samples fall into instruction clusters
    n_clusters = min(50, len(instr_np) // 10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(instr_np)

    # Assign flow samples to clusters
    flow_labels = kmeans.predict(flow_np)

    # Compute distance to cluster centers
    flow_to_center_dist = np.linalg.norm(
        flow_np - kmeans.cluster_centers_[flow_labels], axis=1
    )

    # Reference: average distance of instruction samples to their centers
    instr_labels = kmeans.labels_
    instr_to_center_dist = np.linalg.norm(
        instr_np - kmeans.cluster_centers_[instr_labels], axis=1
    )

    # Cluster overlap: fraction of flow samples within 1.5x instruction radius
    threshold = 1.5 * np.median(instr_to_center_dist)
    cluster_overlap = (flow_to_center_dist < threshold).mean()

    return {
        "mean_mse": float(mean_mse),
        "norm_diff_mean": float(norm_diff),
        "norm_diff_std": float(norm_diff_std),
        "flow_norm_mean": float(flow_norms.mean()),
        "flow_norm_std": float(flow_norms.std()),
        "instr_norm_mean": float(instr_norms.mean()),
        "instr_norm_std": float(instr_norms.std()),
        "cosine_to_centroid": float(cosine_to_centroid),
        "wasserstein_1d": float(wasserstein_1d),
        "cluster_overlap": float(cluster_overlap),
        "n_clusters": n_clusters,
    }


def compute_round_trip_fidelity(embeddings, encoder, decoder, batch_size=16):
    """
    Compute round-trip L2 distance: embed -> decode -> re-encode -> L2.

    L2-r > 0.5 indicates off-manifold embeddings.

    Returns:
    - l2_r_mean: Mean round-trip L2 distance
    - l2_r_std: Std of round-trip L2 distance
    - l2_r_values: List of individual L2-r values
    """
    import torch

    l2_r_values = []
    n_samples = embeddings.shape[0]

    for i in range(0, n_samples, batch_size):
        batch = embeddings[i:i+batch_size]

        # Decode to text
        texts = decoder.decode(batch)

        # Re-encode
        with torch.no_grad():
            re_embedded = encoder.predict(texts, source_lang="eng_Latn")
            if re_embedded.device != batch.device:
                re_embedded = re_embedded.to(batch.device)

        # Compute L2 distance
        l2_dist = (batch - re_embedded).norm(dim=1)
        l2_r_values.extend(l2_dist.cpu().tolist())

        logging.info(f"Round-trip fidelity: {min(i+batch_size, n_samples)}/{n_samples}")

    return {
        "l2_r_mean": float(sum(l2_r_values) / len(l2_r_values)),
        "l2_r_std": float(torch.tensor(l2_r_values).std().item()),
        "l2_r_values": l2_r_values,
    }


def run_decoder_baseline(instruction_texts, instruction_embeddings, encoder, decoder, n_samples=100):
    """
    Verify decoder works on known-good instruction embeddings.

    Tests: original_text -> embed -> decode -> re-embed -> L2-r

    Returns baseline L2-r and sample reconstructions.
    """
    import torch
    import random

    # Sample subset - only use indices with valid text
    valid_indices = [i for i in range(len(instruction_texts)) if instruction_texts[i] is not None]
    indices = random.sample(valid_indices, min(n_samples, len(valid_indices)))

    sample_texts = [instruction_texts[i] for i in indices]
    sample_embeddings = instruction_embeddings[indices]

    # Decode
    decoded_texts = decoder.decode_batch(sample_embeddings, batch_size=16)

    # Re-encode
    with torch.no_grad():
        re_embedded = encoder.predict(decoded_texts, source_lang="eng_Latn")
        if re_embedded.device != sample_embeddings.device:
            re_embedded = re_embedded.to(sample_embeddings.device)

    # L2-r for baseline
    l2_r = (sample_embeddings - re_embedded).norm(dim=1)

    # Sample some reconstructions for inspection
    reconstructions = []
    for i in range(min(5, len(sample_texts))):
        reconstructions.append({
            "original": sample_texts[i],
            "decoded": decoded_texts[i],
            "l2_r": float(l2_r[i].item()),
        })

    return {
        "baseline_l2_r": float(l2_r.mean().item()),
        "baseline_l2_r_std": float(l2_r.std().item()),
        "n_samples": len(indices),
        "reconstructions": reconstructions,
    }


def generate_diagnosis(distribution_metrics, round_trip_flow, round_trip_instr, decoder_baseline):
    """
    Interpret metrics and generate root cause diagnosis.

    Hypotheses:
    - H1: Training data mismatch (flow trained on general text, not instructions)
    - H2: Off-manifold GP guidance (high L2-r for flow samples)
    - H3: SONAR decoder limitations (high baseline L2-r)
    """
    diagnosis = {
        "hypotheses": [],
        "primary_root_cause": None,
        "confidence": None,
        "recommended_fix": None,
        "evidence": {},
    }

    # H1: Training data mismatch
    h1_score = 0
    h1_evidence = []

    if distribution_metrics["mean_mse"] > 0.1:
        h1_score += 2
        h1_evidence.append(f"High MSE to instruction centroid: {distribution_metrics['mean_mse']:.4f}")
    elif distribution_metrics["mean_mse"] > 0.05:
        h1_score += 1
        h1_evidence.append(f"Moderate MSE to instruction centroid: {distribution_metrics['mean_mse']:.4f}")

    if distribution_metrics["cluster_overlap"] < 0.3:
        h1_score += 2
        h1_evidence.append(f"Low cluster overlap: {distribution_metrics['cluster_overlap']:.2%}")
    elif distribution_metrics["cluster_overlap"] < 0.5:
        h1_score += 1
        h1_evidence.append(f"Moderate cluster overlap: {distribution_metrics['cluster_overlap']:.2%}")

    if distribution_metrics["wasserstein_1d"] > 0.5:
        h1_score += 1
        h1_evidence.append(f"High Wasserstein-1D distance: {distribution_metrics['wasserstein_1d']:.4f}")

    diagnosis["hypotheses"].append({
        "id": "H1",
        "name": "Training data mismatch",
        "description": "Flow trained on general text, not instructions",
        "score": h1_score,
        "max_score": 5,
        "evidence": h1_evidence,
        "confidence": "HIGH" if h1_score >= 4 else "MEDIUM" if h1_score >= 2 else "LOW",
    })

    # H2: Off-manifold samples
    h2_score = 0
    h2_evidence = []

    l2_r_flow = round_trip_flow["l2_r_mean"]
    l2_r_instr = round_trip_instr["l2_r_mean"]
    l2_r_ratio = l2_r_flow / l2_r_instr if l2_r_instr > 0 else float("inf")

    if l2_r_flow > 0.5:
        h2_score += 2
        h2_evidence.append(f"High flow L2-r: {l2_r_flow:.4f} (>0.5 threshold)")
    elif l2_r_flow > 0.3:
        h2_score += 1
        h2_evidence.append(f"Moderate flow L2-r: {l2_r_flow:.4f}")

    if l2_r_ratio > 2.0:
        h2_score += 2
        h2_evidence.append(f"Flow L2-r much higher than instructions: {l2_r_ratio:.2f}x")
    elif l2_r_ratio > 1.5:
        h2_score += 1
        h2_evidence.append(f"Flow L2-r higher than instructions: {l2_r_ratio:.2f}x")

    diagnosis["hypotheses"].append({
        "id": "H2",
        "name": "Off-manifold samples",
        "description": "Flow generates embeddings outside SONAR text manifold",
        "score": h2_score,
        "max_score": 4,
        "evidence": h2_evidence,
        "confidence": "HIGH" if h2_score >= 3 else "MEDIUM" if h2_score >= 2 else "LOW",
    })

    # H3: SONAR decoder limitations
    h3_score = 0
    h3_evidence = []

    baseline_l2_r = decoder_baseline["baseline_l2_r"]

    if baseline_l2_r > 0.3:
        h3_score += 2
        h3_evidence.append(f"High baseline L2-r: {baseline_l2_r:.4f} (decoder struggles even with good embeddings)")
    elif baseline_l2_r > 0.15:
        h3_score += 1
        h3_evidence.append(f"Moderate baseline L2-r: {baseline_l2_r:.4f}")
    else:
        h3_evidence.append(f"Low baseline L2-r: {baseline_l2_r:.4f} (decoder works well on good embeddings)")

    diagnosis["hypotheses"].append({
        "id": "H3",
        "name": "SONAR decoder limitations",
        "description": "Decoder struggles with any non-training embeddings",
        "score": h3_score,
        "max_score": 2,
        "evidence": h3_evidence,
        "confidence": "HIGH" if h3_score >= 2 else "MEDIUM" if h3_score >= 1 else "LOW",
    })

    # Determine primary root cause
    hypotheses_by_score = sorted(diagnosis["hypotheses"], key=lambda h: h["score"], reverse=True)
    primary = hypotheses_by_score[0]

    diagnosis["primary_root_cause"] = primary["name"]
    diagnosis["confidence"] = primary["confidence"]

    # Generate recommended fix based on primary cause
    if primary["id"] == "H1":
        diagnosis["recommended_fix"] = (
            "Fine-tune flow model on instruction embeddings. "
            "Create instruction-specific dataset from gsm8k_instructions_vs.pt "
            "and retrain flow model to match instruction distribution."
        )
    elif primary["id"] == "H2":
        diagnosis["recommended_fix"] = (
            "Add round-trip fidelity constraint during guided sampling. "
            "Reduce guidance strength or add L2-r penalty to keep samples on-manifold."
        )
    elif primary["id"] == "H3":
        diagnosis["recommended_fix"] = (
            "Use alternative decoder or apply constrained decoding. "
            "Consider vec2text or fine-tuned decoder for instruction domain."
        )

    # Add evidence summary
    diagnosis["evidence"] = {
        "distribution_mismatch": distribution_metrics["mean_mse"] > 0.05,
        "cluster_overlap": distribution_metrics["cluster_overlap"],
        "l2_r_flow": l2_r_flow,
        "l2_r_instructions": l2_r_instr,
        "l2_r_ratio": l2_r_ratio,
        "decoder_baseline_l2_r": baseline_l2_r,
    }

    return diagnosis


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Lazy imports
    import torch
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    from src.ecoflow.decoder import SonarDecoder

    device = args.device
    logging.info(f"Device: {device}")
    logging.info(f"N samples: {args.n_samples}")

    # Find flow checkpoint
    checkpoint_path = args.flow_checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_flow_checkpoint()
    logging.info(f"Flow checkpoint: {checkpoint_path}")

    # Load instruction data
    logging.info(f"Loading instruction data: {args.instruction_data}")
    instr_data = torch.load(args.instruction_data, map_location="cpu", weights_only=False)
    instruction_embeddings = instr_data["embeddings"].to(device)
    instruction_texts = instr_data["instructions"]
    logging.info(f"Loaded {len(instruction_texts)} instructions, embeddings: {instruction_embeddings.shape}")

    # Load flow model
    flow_model = load_flow_model(checkpoint_path, device)

    # Generate flow samples
    logging.info(f"Generating {args.n_samples} flow samples (no guidance)...")
    flow_samples = flow_model.sample(
        n_samples=args.n_samples,
        device=device,
        method="heun",
        num_steps=50,
        denormalize=True,
    )
    logging.info(f"Generated flow samples: {flow_samples.shape}")
    logging.info(f"Flow sample stats: mean={flow_samples.mean():.4f}, std={flow_samples.std():.4f}, L2={flow_samples.norm(dim=1).mean():.4f}")

    # Initialize SONAR encoder/decoder
    logging.info("Initializing SONAR encoder...")
    encoder = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=torch.device(device),
    )

    logging.info("Initializing SONAR decoder...")
    decoder = SonarDecoder(device=device, ngram_block_size=3)

    # ============================================================
    # Analysis 1: Distribution Comparison
    # ============================================================
    logging.info("=" * 60)
    logging.info("Analysis 1: Distribution Comparison")
    logging.info("=" * 60)

    distribution_metrics = compute_distribution_metrics(flow_samples, instruction_embeddings)
    logging.info(f"Mean MSE to instruction centroid: {distribution_metrics['mean_mse']:.4f}")
    logging.info(f"L2 norm - Flow: {distribution_metrics['flow_norm_mean']:.4f} +/- {distribution_metrics['flow_norm_std']:.4f}")
    logging.info(f"L2 norm - Instr: {distribution_metrics['instr_norm_mean']:.4f} +/- {distribution_metrics['instr_norm_std']:.4f}")
    logging.info(f"Cosine to centroid: {distribution_metrics['cosine_to_centroid']:.4f}")
    logging.info(f"Wasserstein-1D: {distribution_metrics['wasserstein_1d']:.4f}")
    logging.info(f"Cluster overlap: {distribution_metrics['cluster_overlap']:.2%}")

    # ============================================================
    # Analysis 2: Round-Trip Fidelity
    # ============================================================
    logging.info("=" * 60)
    logging.info("Analysis 2: Round-Trip Fidelity")
    logging.info("=" * 60)

    # Sample subset for round-trip (expensive operation)
    n_roundtrip = min(100, args.n_samples)

    logging.info(f"Computing round-trip fidelity for {n_roundtrip} flow samples...")
    round_trip_flow = compute_round_trip_fidelity(
        flow_samples[:n_roundtrip], encoder, decoder, batch_size=16
    )
    logging.info(f"Flow L2-r: {round_trip_flow['l2_r_mean']:.4f} +/- {round_trip_flow['l2_r_std']:.4f}")

    logging.info(f"Computing round-trip fidelity for {n_roundtrip} instruction embeddings...")
    round_trip_instr = compute_round_trip_fidelity(
        instruction_embeddings[:n_roundtrip], encoder, decoder, batch_size=16
    )
    logging.info(f"Instruction L2-r: {round_trip_instr['l2_r_mean']:.4f} +/- {round_trip_instr['l2_r_std']:.4f}")

    # ============================================================
    # Analysis 3: SONAR Decoder Baseline
    # ============================================================
    logging.info("=" * 60)
    logging.info("Analysis 3: SONAR Decoder Baseline")
    logging.info("=" * 60)

    decoder_baseline = run_decoder_baseline(
        instruction_texts, instruction_embeddings, encoder, decoder, n_samples=100
    )
    logging.info(f"Baseline L2-r: {decoder_baseline['baseline_l2_r']:.4f} +/- {decoder_baseline['baseline_l2_r_std']:.4f}")

    logging.info("Sample reconstructions:")
    for i, recon in enumerate(decoder_baseline["reconstructions"][:3]):
        orig = recon['original'] or "(none)"
        decoded = recon['decoded'] or "(none)"
        logging.info(f"  [{i+1}] Original: {orig[:80]}...")
        logging.info(f"      Decoded:  {decoded[:80]}...")
        logging.info(f"      L2-r: {recon['l2_r']:.4f}")

    # ============================================================
    # Generate Diagnosis
    # ============================================================
    logging.info("=" * 60)
    logging.info("Generating Diagnosis")
    logging.info("=" * 60)

    diagnosis = generate_diagnosis(
        distribution_metrics,
        {"l2_r_mean": round_trip_flow["l2_r_mean"], "l2_r_std": round_trip_flow["l2_r_std"]},
        {"l2_r_mean": round_trip_instr["l2_r_mean"], "l2_r_std": round_trip_instr["l2_r_std"]},
        decoder_baseline,
    )

    logging.info(f"Primary root cause: {diagnosis['primary_root_cause']}")
    logging.info(f"Confidence: {diagnosis['confidence']}")
    logging.info(f"Recommended fix: {diagnosis['recommended_fix']}")

    for h in diagnosis["hypotheses"]:
        logging.info(f"  {h['id']}: {h['name']} - Score {h['score']}/{h['max_score']} ({h['confidence']})")
        for e in h["evidence"]:
            logging.info(f"    - {e}")

    # ============================================================
    # Build Report
    # ============================================================
    report = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "flow_checkpoint": checkpoint_path,
            "instruction_data": args.instruction_data,
            "n_samples": args.n_samples,
            "device": device,
        },
        "distribution_comparison": distribution_metrics,
        "round_trip_fidelity": {
            "l2_r_flow_mean": round_trip_flow["l2_r_mean"],
            "l2_r_flow_std": round_trip_flow["l2_r_std"],
            "l2_r_instructions_mean": round_trip_instr["l2_r_mean"],
            "l2_r_instructions_std": round_trip_instr["l2_r_std"],
            "l2_r_ratio": round_trip_flow["l2_r_mean"] / round_trip_instr["l2_r_mean"] if round_trip_instr["l2_r_mean"] > 0 else float("inf"),
            "n_samples_tested": n_roundtrip,
        },
        "decoder_baseline": {
            "baseline_l2_r": decoder_baseline["baseline_l2_r"],
            "baseline_l2_r_std": decoder_baseline["baseline_l2_r_std"],
            "n_samples": decoder_baseline["n_samples"],
            "sample_reconstructions": decoder_baseline["reconstructions"],
        },
        "diagnosis": diagnosis,
    }

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logging.info(f"\nDiagnostic report saved to: {output_path}")
    logging.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
