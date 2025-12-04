"""
Gradient-based Prompt Optimization for Maximum Expected Improvement.

Optimizes BERT embeddings via gradient ascent to maximize EI,
then uses vec2text to convert optimized embeddings back to text.
"""
import json
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from hbbops.hbbops import HbBoPs, Prompt
from hbbops.run_hbbops import load_instructions, load_exemplars


def compute_ei_differentiable(embedding_1536d: torch.Tensor,
                               feature_extractor: nn.Module,
                               train_latents: torch.Tensor,
                               train_y: torch.Tensor,
                               kernel_lengthscale: torch.Tensor,
                               kernel_outputscale: torch.Tensor,
                               noise_var: torch.Tensor,
                               X_min: torch.Tensor,
                               X_max: torch.Tensor,
                               y_mean: torch.Tensor,
                               y_std: torch.Tensor,
                               vmin_b: float) -> torch.Tensor:
    """
    Compute Expected Improvement in a differentiable way.

    Uses manual GP prediction to enable gradient flow.

    Pipeline:
    embedding_1536d → normalize → FeatureExtractor → kernel computation → mean, std → EI

    Args:
        embedding_1536d: torch.Tensor (1536,) with requires_grad=True
        feature_extractor: trained MLP
        train_latents: (N, 10) latent features of training data
        train_y: (N,) standardized training targets
        kernel params: lengthscale, outputscale, noise variance
        X_min, X_max: input normalization parameters
        y_mean, y_std: output standardization parameters
        vmin_b: best observed error (float)

    Returns:
        ei: torch.Tensor (scalar) - Expected Improvement (to maximize)
    """
    # 1. Normalize input to unit cube
    denom = X_max - X_min
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    X_norm = (embedding_1536d - X_min) / denom

    # 2. Extract latent features via FeatureExtractor
    inst_norm = X_norm[:768].unsqueeze(0)
    ex_norm = X_norm[768:].unsqueeze(0)
    test_latent = feature_extractor(inst_norm, ex_norm)  # (1, 10)

    # 3. Compute kernel matrices (Matérn 5/2)
    # K(x, x') = outputscale * matern52(dist / lengthscale)
    def matern52_kernel(x1, x2, lengthscale, outputscale):
        # ARD Matern 5/2: k(r) = (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
        # where r = ||x1 - x2|| / lengthscale (with ARD)
        diff = x1.unsqueeze(1) - x2.unsqueeze(0)  # (N1, N2, D)
        scaled_diff = diff / lengthscale  # ARD scaling
        dist = torch.sqrt((scaled_diff ** 2).sum(dim=-1) + 1e-8)  # (N1, N2)
        sqrt5 = torch.sqrt(torch.tensor(5.0, device=x1.device))
        k = outputscale * (1 + sqrt5 * dist + 5.0/3.0 * dist**2) * torch.exp(-sqrt5 * dist)
        return k

    # K_train_train: (N, N)
    K_train = matern52_kernel(train_latents, train_latents, kernel_lengthscale, kernel_outputscale)
    K_train = K_train + noise_var * torch.eye(K_train.shape[0], device=K_train.device)

    # K_test_train: (1, N)
    K_test_train = matern52_kernel(test_latent, train_latents, kernel_lengthscale, kernel_outputscale)

    # K_test_test: (1, 1)
    K_test = matern52_kernel(test_latent, test_latent, kernel_lengthscale, kernel_outputscale)

    # 4. GP predictive mean and variance
    # mean = K_test_train @ K_train^{-1} @ y
    # var = K_test_test - K_test_train @ K_train^{-1} @ K_test_train^T
    L = torch.linalg.cholesky(K_train + 1e-4 * torch.eye(K_train.shape[0], device=K_train.device))
    alpha = torch.cholesky_solve(train_y.unsqueeze(1), L)  # K^{-1} @ y
    mean_norm = (K_test_train @ alpha).squeeze()  # standardized mean

    v = torch.cholesky_solve(K_test_train.T, L)  # K^{-1} @ K_test_train^T
    var_norm = (K_test - K_test_train @ v).squeeze()  # standardized variance
    std_norm = torch.sqrt(torch.clamp(var_norm, min=1e-8))

    # 5. De-standardize
    mean = mean_norm * y_std + y_mean  # predicted error
    std = std_norm * y_std

    # 6. EI formula: EI = (vmin - μ) * Φ(z) + σ * φ(z)
    z = (vmin_b - mean) / (std + 1e-8)

    # Use torch distributions for differentiability
    normal = torch.distributions.Normal(
        torch.zeros(1, device=embedding_1536d.device),
        torch.ones(1, device=embedding_1536d.device)
    )
    cdf_z = normal.cdf(z)
    pdf_z = torch.exp(normal.log_prob(z))

    ei = (vmin_b - mean) * cdf_z + std * pdf_z

    return ei.squeeze()


def optimize_embedding_for_max_ei(
    feature_extractor: nn.Module,
    train_latents: torch.Tensor,
    train_y: torch.Tensor,
    kernel_lengthscale: torch.Tensor,
    kernel_outputscale: torch.Tensor,
    noise_var: torch.Tensor,
    X_min: torch.Tensor,
    X_max: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    init_embedding: torch.Tensor,
    vmin_b: float,
    device: torch.device,
    n_steps: int = 500,
    lr: float = 0.01,
    verbose: bool = True
) -> Tuple[torch.Tensor, float]:
    """
    Gradient ASCENT to maximize Expected Improvement.

    Args:
        feature_extractor: trained MLP (FeatureExtractor)
        train_latents: (N, 10) latent features of training data
        train_y: (N,) standardized training targets
        kernel params: from trained GP
        X_min, X_max: input normalization
        y_mean, y_std: output standardization
        init_embedding: starting point (1536D)
        vmin_b: best observed error
        device: torch device
        n_steps: optimization steps
        lr: learning rate
        verbose: print progress

    Returns:
        best_embedding: optimized 1536D embedding
        best_ei: achieved EI value
    """
    # Clone and enable gradients
    embedding = init_embedding.clone().to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([embedding], lr=lr)

    # Pre-compute Cholesky for efficiency (train data doesn't change)
    # Actually, we'll compute it inside compute_ei for simplicity

    best_ei = -float('inf')
    best_embedding = None
    history = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # Compute EI
        ei = compute_ei_differentiable(
            embedding,
            feature_extractor,
            train_latents,
            train_y,
            kernel_lengthscale,
            kernel_outputscale,
            noise_var,
            X_min,
            X_max,
            y_mean,
            y_std,
            vmin_b
        )

        # Gradient ASCENT: loss = -EI
        loss = -ei
        loss.backward()
        optimizer.step()

        # Track best
        ei_val = ei.item()
        if ei_val > best_ei:
            best_ei = ei_val
            best_embedding = embedding.detach().clone()

        history.append(ei_val)

        if verbose and step % 100 == 0:
            print(f"Step {step}: EI = {ei_val:.6f}")

    if verbose:
        print(f"Final: EI = {best_ei:.6f} (improvement: {best_ei - history[0]:.6f})")

    return best_embedding, best_ei


def find_nearest_text(
    embedding: torch.Tensor,
    all_embeddings: np.ndarray,
    texts: list,
    k: int = 1
) -> list:
    """Find k nearest texts to the given embedding."""
    emb_np = embedding.cpu().numpy()
    distances = np.linalg.norm(all_embeddings - emb_np, axis=1)
    nearest_idx = np.argsort(distances)[:k]
    return [(texts[i], distances[i]) for i in nearest_idx]


def embedding_to_text_nearest(
    optimized_embedding: torch.Tensor,
    instruction_embeddings: dict,
    exemplar_embeddings: dict,
    instructions: list,
    exemplars: list
) -> Tuple[str, str, dict]:
    """
    Convert optimized embedding to text via nearest neighbor.

    Args:
        optimized_embedding: (1536D) tensor
        instruction_embeddings: dict {id: 768D array}
        exemplar_embeddings: dict {id: 768D array}
        instructions: list of instruction texts
        exemplars: list of exemplar texts

    Returns:
        instruction_text, exemplar_text, metadata
    """
    inst_emb = optimized_embedding[:768].cpu().numpy()
    ex_emb = optimized_embedding[768:].cpu().numpy()

    # Build arrays for nearest neighbor search
    inst_ids = list(instruction_embeddings.keys())
    inst_arr = np.array([instruction_embeddings[i] for i in inst_ids])

    ex_ids = list(exemplar_embeddings.keys())
    ex_arr = np.array([exemplar_embeddings[i] for i in ex_ids])

    # Find nearest
    inst_distances = np.linalg.norm(inst_arr - inst_emb, axis=1)
    nearest_inst_idx = np.argmin(inst_distances)
    nearest_inst_id = inst_ids[nearest_inst_idx]

    ex_distances = np.linalg.norm(ex_arr - ex_emb, axis=1)
    nearest_ex_idx = np.argmin(ex_distances)
    nearest_ex_id = ex_ids[nearest_ex_idx]

    metadata = {
        'instruction_id': nearest_inst_id,
        'exemplar_id': nearest_ex_id,
        'instruction_distance': float(inst_distances[nearest_inst_idx]),
        'exemplar_distance': float(ex_distances[nearest_ex_idx])
    }

    return instructions[nearest_inst_id], exemplars[nearest_ex_id], metadata


def embedding_to_text_vec2text(
    optimized_embedding: torch.Tensor,
    corrector=None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert optimized embedding to text using vec2text.

    Vec2text inverts embeddings back to text using a trained corrector model.

    NOTE: Vec2text models are trained for specific encoders (GTR, OpenAI Ada, etc.).
    Our HbBoPs uses BERT embeddings, which are in a different embedding space.
    The inversion quality may be limited due to this mismatch.

    Args:
        optimized_embedding: (1536D) tensor
        corrector: vec2text Corrector instance (loaded if None)

    Returns:
        instruction_text, exemplar_text (or None if inversion fails)
    """
    try:
        import vec2text
    except ImportError:
        print("vec2text not available. Install with: uv add vec2text")
        return None, None

    inst_emb = optimized_embedding[:768].cpu()
    ex_emb = optimized_embedding[768:].cpu()

    # Load corrector if not provided
    if corrector is None:
        print("Loading vec2text corrector (GTR-base)...")
        try:
            # Use ielabgroup models which are compatible with sentence-transformers/gtr-t5-base
            # These use the same mean pooling as our PromptEncoder
            inversion_model = vec2text.models.InversionModel.from_pretrained(
                "ielabgroup/vec2text_gtr-base-st_inversion"
            )
            corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained(
                "ielabgroup/vec2text_gtr-base-st_corrector"
            )
            corrector = vec2text.load_corrector(inversion_model, corrector_model)
        except Exception as e:
            print(f"Failed to load vec2text corrector: {e}")
            print("Trying alternative jxm models...")
            try:
                # Fallback to jxm models
                inversion_model = vec2text.models.InversionModel.from_pretrained(
                    "jxm/gtr__nq__32"
                )
                corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained(
                    "jxm/gtr__nq__32__correct"
                )
                corrector = vec2text.load_corrector(inversion_model, corrector_model)
            except Exception as e2:
                print(f"All vec2text models failed: {e2}")
                print("The GTR-based models require ~2GB download from HuggingFace.")
                return None, None

    # Invert embeddings using vec2text API
    print("Inverting instruction embedding...")
    instruction_text = None
    try:
        # vec2text.invert_embeddings expects embeddings on GPU if available
        device = next(corrector.model.parameters()).device
        inst_tensor = inst_emb.unsqueeze(0).to(device)
        results = vec2text.invert_embeddings(
            embeddings=inst_tensor,
            corrector=corrector,
            num_steps=20,
            sequence_beam_width=4
        )
        instruction_text = results[0] if results else None
    except Exception as e:
        print(f"Instruction inversion failed: {e}")

    print("Inverting exemplar embedding...")
    exemplar_text = None
    try:
        device = next(corrector.model.parameters()).device
        ex_tensor = ex_emb.unsqueeze(0).to(device)
        results = vec2text.invert_embeddings(
            embeddings=ex_tensor,
            corrector=corrector,
            num_steps=20,
            sequence_beam_width=4
        )
        exemplar_text = results[0] if results else None
    except Exception as e:
        print(f"Exemplar inversion failed: {e}")

    return instruction_text, exemplar_text


def main():
    """Run gradient-based prompt optimization."""
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    datasets_dir = base_dir.parent / "datasets" / "hbbops"
    full_grid_path = datasets_dir / "full_grid_combined.jsonl"

    print("Loading data...")
    instructions = load_instructions(str(datasets_dir / "instructions.txt"))
    exemplars = load_exemplars(str(datasets_dir / "examples.txt"))

    with open(data_dir / "validation.json", 'r') as f:
        validation_data = json.load(f)
    n_valid = len(validation_data)

    # Initialize HbBoPs
    print("Initializing HbBoPs...")
    hbbops = HbBoPs(
        instructions=instructions,
        exemplars=exemplars,
        validation_data=validation_data,
        llm_evaluator=lambda p, d: 0.0,
        device="auto"
    )

    # Load results and train GP
    print("Loading results and training GP...")
    results = []
    with open(full_grid_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    id_to_idx = {(p.instruction_id, p.exemplar_id): idx for idx, p in enumerate(hbbops.prompts)}

    # Populate design data
    all_embeddings = []
    all_errors = []
    for res in results:
        if (res['instruction_id'], res['exemplar_id']) not in id_to_idx:
            continue
        p_idx = id_to_idx[(res['instruction_id'], res['exemplar_id'])]
        prompt = hbbops.prompts[p_idx]
        inst_emb, ex_emb = hbbops.embed_prompt(prompt)
        hbbops.design_data.append((p_idx, inst_emb, ex_emb, res['error_rate'], n_valid))
        all_embeddings.append(np.concatenate([inst_emb, ex_emb]))
        all_errors.append(res['error_rate'])

    all_embeddings = np.array(all_embeddings)
    all_errors = np.array(all_errors)

    # Train GP
    print("Training GP...")
    hbbops.train_gp(fidelity=n_valid, min_observations=10)

    # Extract GP parameters for differentiable computation
    device = hbbops.device
    feature_extractor = hbbops.gp_model.feature_extractor

    # Get kernel parameters
    kernel_lengthscale = hbbops.gp_model.covar_module.base_kernel.lengthscale.detach()
    kernel_outputscale = hbbops.gp_model.covar_module.outputscale.detach()
    noise_var = hbbops.likelihood.noise.detach()

    print(f"Kernel lengthscale shape: {kernel_lengthscale.shape}")
    print(f"Kernel outputscale: {kernel_outputscale.item():.4f}")
    print(f"Noise variance: {noise_var.item():.6f}")

    # Compute training latents
    print("Computing training latent features...")
    X_train = torch.tensor(all_embeddings, dtype=torch.float32, device=device)
    denom = hbbops.X_max - hbbops.X_min
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    X_train_norm = (X_train - hbbops.X_min) / denom

    with torch.no_grad():
        inst_norm = X_train_norm[:, :768]
        ex_norm = X_train_norm[:, 768:]
        train_latents = feature_extractor(inst_norm, ex_norm)

    # Standardized training targets
    train_y = torch.tensor(all_errors, dtype=torch.float32, device=device)
    train_y_norm = (train_y - hbbops.y_mean) / hbbops.y_std

    print(f"Train latents shape: {train_latents.shape}")
    print(f"Train y shape: {train_y_norm.shape}")

    # Find best observed error
    vmin_b = float(all_errors.min())
    print(f"Best observed error (vmin_b): {vmin_b:.4f}")

    # Initialize from best prompt
    best_idx = np.argmin(all_errors)
    init_embedding = torch.tensor(all_embeddings[best_idx], dtype=torch.float32)
    print(f"Initializing from prompt {best_idx} with error {all_errors[best_idx]:.4f}")

    # Compute initial EI
    with torch.no_grad():
        init_ei = compute_ei_differentiable(
            init_embedding.to(device),
            feature_extractor,
            train_latents,
            train_y_norm,
            kernel_lengthscale.squeeze(),
            kernel_outputscale,
            noise_var,
            hbbops.X_min,
            hbbops.X_max,
            hbbops.y_mean,
            hbbops.y_std,
            vmin_b
        )
    print(f"Initial EI: {init_ei.item():.6f}")

    # Optimize embedding
    print("\n" + "="*60)
    print("Starting gradient ascent optimization...")
    print("="*60)

    optimized_embedding, best_ei = optimize_embedding_for_max_ei(
        feature_extractor,
        train_latents,
        train_y_norm,
        kernel_lengthscale.squeeze(),
        kernel_outputscale,
        noise_var,
        hbbops.X_min,
        hbbops.X_max,
        hbbops.y_mean,
        hbbops.y_std,
        init_embedding,
        vmin_b,
        device,
        n_steps=500,
        lr=0.01,
        verbose=True
    )

    print(f"\nOptimization complete!")
    print(f"Initial EI: {init_ei.item():.6f}")
    print(f"Final EI: {best_ei:.6f}")
    print(f"Improvement: {best_ei - init_ei.item():.6f}")

    # Convert to text via nearest neighbor
    print("\n" + "="*60)
    print("Converting optimized embedding to text...")
    print("="*60)

    instruction_text, exemplar_text, metadata = embedding_to_text_nearest(
        optimized_embedding,
        hbbops.instruction_embeddings,
        hbbops.exemplar_embeddings,
        instructions,
        exemplars
    )

    print(f"\nNearest instruction (id={metadata['instruction_id']}, dist={metadata['instruction_distance']:.4f}):")
    print(f"  {instruction_text}")
    print(f"\nNearest exemplar (id={metadata['exemplar_id']}, dist={metadata['exemplar_distance']:.4f}):")
    print(f"  {exemplar_text[:200]}...")

    # Verify by re-embedding
    print("\n" + "="*60)
    print("Verifying reconstructed prompt...")
    print("="*60)

    # Get embedding of the nearest prompt
    nearest_inst_emb = hbbops.instruction_embeddings[metadata['instruction_id']]
    nearest_ex_emb = hbbops.exemplar_embeddings[metadata['exemplar_id']]
    reconstructed_emb = torch.tensor(
        np.concatenate([nearest_inst_emb, nearest_ex_emb]),
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        reconstructed_ei = compute_ei_differentiable(
            reconstructed_emb,
            feature_extractor,
            train_latents,
            train_y_norm,
            kernel_lengthscale.squeeze(),
            kernel_outputscale,
            noise_var,
            hbbops.X_min,
            hbbops.X_max,
            hbbops.y_mean,
            hbbops.y_std,
            vmin_b
        )

    print(f"Reconstructed prompt EI: {reconstructed_ei.item():.6f}")
    print(f"Optimized embedding EI: {best_ei:.6f}")
    print(f"Gap (due to discretization): {best_ei - reconstructed_ei.item():.6f}")

    # Try vec2text inversion
    print("\n" + "="*60)
    print("Attempting vec2text inversion...")
    print("="*60)

    vec2text_instruction, vec2text_exemplar = embedding_to_text_vec2text(optimized_embedding)

    if vec2text_instruction:
        print(f"\nVec2text generated instruction:")
        print(f"  {vec2text_instruction}")
    if vec2text_exemplar:
        print(f"\nVec2text generated exemplar:")
        print(f"  {vec2text_exemplar[:300]}...")

    # Save results
    output = {
        'vmin_b': vmin_b,
        'initial_ei': float(init_ei.item()),
        'optimized_ei': best_ei,
        'reconstructed_ei': float(reconstructed_ei.item()),
        'nearest_instruction_id': metadata['instruction_id'],
        'nearest_exemplar_id': metadata['exemplar_id'],
        'instruction_distance': metadata['instruction_distance'],
        'exemplar_distance': metadata['exemplar_distance'],
        'instruction_text': instruction_text,
        'exemplar_text': exemplar_text[:500],
        'vec2text_instruction': vec2text_instruction,
        'vec2text_exemplar': vec2text_exemplar[:500] if vec2text_exemplar else None,
        # Save embeddings for visualization
        'optimized_embedding': optimized_embedding.cpu().tolist(),
        'init_embedding': init_embedding.tolist()
    }

    output_path = results_dir / "prompt_inverse_result.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
