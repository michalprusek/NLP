
import json
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hbbops.hbbops import HbBoPs
from hbbops.run_hbbops import load_instructions, load_exemplars


def visualize_mlp_projections(hbbops, all_results, id_to_idx, results_dir):
    """
    Create visualization comparing BERT embeddings with different projections.

    Uses the TRAINED FeatureExtractor from Deep Kernel GP (end-to-end trained
    on accuracy prediction) to extract learned MLP representations.

    Grid layout (2x3):
    - Row 1: Instructions (BERT→UMAP, BERT→PCA(32D)→UMAP, BERT→Trained MLP(768→32)→UMAP)
    - Row 2: Exemplars (same 3 transformations)
    """
    print("\n=== Creating MLP projection visualization (using trained encoders) ===")

    # Get the trained feature extractor from GP model
    feature_extractor = hbbops.gp_model.feature_extractor
    feature_extractor.eval()

    # Extract the trained instruction and exemplar encoders (768 -> 64 -> 32)
    trained_inst_encoder = feature_extractor.instruction_encoder
    trained_ex_encoder = feature_extractor.exemplar_encoder

    # Extract separate embeddings for instructions and exemplars
    inst_embeddings = []
    ex_embeddings = []
    accuracies = []

    for res in all_results:
        p_idx = id_to_idx[(res['instruction_id'], res['exemplar_id'])]
        prompt = hbbops.prompts[p_idx]
        inst_emb, ex_emb = hbbops.embed_prompt(prompt)
        inst_embeddings.append(inst_emb)
        ex_embeddings.append(ex_emb)
        accuracies.append(1.0 - res['error_rate'])

    X_inst = np.array(inst_embeddings)  # (N, 768)
    X_ex = np.array(ex_embeddings)      # (N, 768)
    accuracies = np.array(accuracies)

    print(f"Instructions shape: {X_inst.shape}")
    print(f"Exemplars shape: {X_ex.shape}")

    # Normalize inputs using the same normalization as GP training
    X_inst_tensor = torch.tensor(X_inst, dtype=torch.float32, device=hbbops.device)
    X_ex_tensor = torch.tensor(X_ex, dtype=torch.float32, device=hbbops.device)

    # Apply normalization (X_mean and X_std are 1D tensors for concatenated [inst, ex])
    input_dim = 768
    inst_mean = hbbops.X_mean[:input_dim]
    inst_std = hbbops.X_std[:input_dim]
    ex_mean = hbbops.X_mean[input_dim:]
    ex_std = hbbops.X_std[input_dim:]

    X_inst_norm = (X_inst_tensor - inst_mean) / inst_std
    X_ex_norm = (X_ex_tensor - ex_mean) / ex_std

    # Compute all projections
    print("Computing projections...")

    # Instructions projections
    # 1. BERT -> UMAP (direct)
    reducer_inst_direct = umap.UMAP(random_state=42)
    emb_inst_direct = reducer_inst_direct.fit_transform(X_inst)

    # 2. BERT -> PCA(32D) -> UMAP
    pca_inst = PCA(n_components=32)
    X_inst_pca = pca_inst.fit_transform(X_inst)
    reducer_inst_pca = umap.UMAP(random_state=42)
    emb_inst_pca = reducer_inst_pca.fit_transform(X_inst_pca)

    # 3. BERT -> Trained MLP(768->32) -> UMAP
    with torch.no_grad():
        X_inst_mlp = trained_inst_encoder(X_inst_norm).cpu().numpy()
    print(f"Trained instruction encoder output shape: {X_inst_mlp.shape}")
    reducer_inst_mlp = umap.UMAP(random_state=42)
    emb_inst_mlp = reducer_inst_mlp.fit_transform(X_inst_mlp)

    # Exemplars projections
    # 1. BERT -> UMAP (direct)
    reducer_ex_direct = umap.UMAP(random_state=42)
    emb_ex_direct = reducer_ex_direct.fit_transform(X_ex)

    # 2. BERT -> PCA(32D) -> UMAP
    pca_ex = PCA(n_components=32)
    X_ex_pca = pca_ex.fit_transform(X_ex)
    reducer_ex_pca = umap.UMAP(random_state=42)
    emb_ex_pca = reducer_ex_pca.fit_transform(X_ex_pca)

    # 3. BERT -> Trained MLP(768->32) -> UMAP
    with torch.no_grad():
        X_ex_mlp = trained_ex_encoder(X_ex_norm).cpu().numpy()
    print(f"Trained exemplar encoder output shape: {X_ex_mlp.shape}")
    reducer_ex_mlp = umap.UMAP(random_state=42)
    emb_ex_mlp = reducer_ex_mlp.fit_transform(X_ex_mlp)

    # Plotting
    print("Plotting...")
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    def plot_scatter(ax, embedding, title):
        sc = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=accuracies,
            cmap='viridis',
            s=50,
            alpha=0.8,
            vmin=0.0, vmax=1.0
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        return sc

    # Row 1: Instructions
    plot_scatter(axes[0, 0], emb_inst_direct, 'Instructions: BERT → UMAP')
    plot_scatter(axes[0, 1], emb_inst_pca, 'Instructions: BERT → PCA(32D) → UMAP')
    plot_scatter(axes[0, 2], emb_inst_mlp, 'Instructions: BERT → Trained MLP → UMAP')

    # Row 2: Exemplars
    plot_scatter(axes[1, 0], emb_ex_direct, 'Exemplars: BERT → UMAP')
    plot_scatter(axes[1, 1], emb_ex_pca, 'Exemplars: BERT → PCA(32D) → UMAP')
    sc = plot_scatter(axes[1, 2], emb_ex_mlp, 'Exemplars: BERT → Trained MLP → UMAP')

    # Colorbar (shared)
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label='Accuracy')

    output_path = results_dir / "mlp_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"MLP visualization saved to {output_path}")


def main():
    # Paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    full_grid_path = results_dir / "full_grid_combined.jsonl"
    
    print(f"Loading data from {full_grid_path}")
    
    # Load instructions and exemplars
    instructions = load_instructions(str(base_dir / "instructions.txt"))
    exemplars = load_exemplars(str(base_dir / "examples.txt"))
    
    # Load validation data
    with open(data_dir / "validation.json", 'r') as f:
        validation_data = json.load(f)
    n_valid = len(validation_data)
    
    # Mock LLM evaluator
    def mock_evaluator(prompt, data):
        return 0.0
        
    # Initialize HbBoPs
    print("Initializing HbBoPs...")
    hbbops = HbBoPs(
        instructions=instructions,
        exemplars=exemplars,
        validation_data=validation_data,
        llm_evaluator=mock_evaluator,
        device="auto"
    )
    
    # Load results
    print("Loading full grid results...")
    results = []
    with open(full_grid_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
            
    # Create map for quick lookup
    id_to_idx = {}
    for idx, p in enumerate(hbbops.prompts):
        id_to_idx[(p.instruction_id, p.exemplar_id)] = idx
        
    # Filter valid results
    valid_results = [r for r in results if (r['instruction_id'], r['exemplar_id']) in id_to_idx]
    
    # Split into Train (80%) and Test (20%)
    print("Splitting data 80/20...")
    train_results, test_results = train_test_split(valid_results, test_size=0.2, random_state=42)
    
    print(f"Train size: {len(train_results)}")
    print(f"Test size: {len(test_results)}")
    
    # 1. Train Deep Kernel GP on Train Set ONLY
    print("Populating design data with TRAIN set only...")
    hbbops.design_data = [] # Clear existing
    
    for res in train_results:
        p_idx = id_to_idx[(res['instruction_id'], res['exemplar_id'])]
        prompt = hbbops.prompts[p_idx]
        inst_emb, ex_emb = hbbops.embed_prompt(prompt)
        hbbops.design_data.append((p_idx, inst_emb, ex_emb, res['error_rate'], n_valid))
        
    print("Training Deep Kernel GP on Train set...")
    hbbops.train_gp(fidelity=n_valid, min_observations=10)
    
    # 2. Extract Features and Embeddings for ALL data (Train + Test)
    print("Extracting features for all data...")
    hbbops.gp_model.eval()
    
    # We need to process train and test separately to keep track, 
    # but for UMAP it's best to fit on all or fit on train. 
    # Let's fit UMAP on ALL to have a consistent coordinate system for visualization.
    
    all_results = train_results + test_results
    train_indices = list(range(len(train_results)))
    test_indices = list(range(len(train_results), len(all_results)))
    
    dk_features = []
    bert_embeddings = []
    accuracies = []
    
    for res in all_results:
        p_idx = id_to_idx[(res['instruction_id'], res['exemplar_id'])]
        prompt = hbbops.prompts[p_idx]
        
        # BERT
        inst_emb, ex_emb = hbbops.embed_prompt(prompt)
        combined_emb = np.concatenate([inst_emb, ex_emb])
        bert_embeddings.append(combined_emb)
        
        # Deep Kernel
        inst_tensor = torch.tensor(inst_emb, dtype=torch.float32, device=hbbops.device).unsqueeze(0)
        ex_tensor = torch.tensor(ex_emb, dtype=torch.float32, device=hbbops.device).unsqueeze(0)
        
        X_input = torch.cat([inst_tensor, ex_tensor], dim=1)
        X_norm = (X_input - hbbops.X_mean) / hbbops.X_std
        
        with torch.no_grad():
            input_dim = 768
            inst_norm = X_norm[:, :input_dim]
            ex_norm = X_norm[:, input_dim:]
            latent = hbbops.gp_model.feature_extractor(inst_norm, ex_norm)
            dk_features.append(latent.cpu().numpy().squeeze())
            
        accuracies.append(1.0 - res['error_rate'])
        
    X_bert = np.array(bert_embeddings)
    X_dk = np.array(dk_features)
    accuracies = np.array(accuracies)
    
    # Compute Projections (Fit on ALL for shared space)
    print("Computing projections...")
    
    # Deep Kernel -> UMAP
    reducer_dk = umap.UMAP(random_state=42)
    embedding_dk = reducer_dk.fit_transform(X_dk)
    
    # BERT -> PCA -> UMAP
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_bert)
    reducer_pca = umap.UMAP(random_state=42)
    embedding_pca = reducer_pca.fit_transform(X_pca)
    
    # BERT -> UMAP
    reducer_direct = umap.UMAP(random_state=42)
    embedding_direct = reducer_direct.fit_transform(X_bert)
    
    # Plotting
    print("Plotting...")
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    
    # Helper to plot
    def plot_scatter(ax, embedding, indices, title):
        sc = ax.scatter(
            embedding[indices, 0],
            embedding[indices, 1],
            c=accuracies[indices],
            cmap='viridis',
            s=50,
            alpha=0.8,
            vmin=0.0, vmax=1.0 # Fixed scale for comparison
        )
        ax.set_title(title)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        return sc

    # Row 1: Train
    plot_scatter(axes[0, 0], embedding_dk, train_indices, 'Train: Deep Kernel -> UMAP')
    plot_scatter(axes[0, 1], embedding_pca, train_indices, 'Train: BERT -> PCA -> UMAP')
    sc = plot_scatter(axes[0, 2], embedding_direct, train_indices, 'Train: BERT -> UMAP')
    
    # Row 2: Test
    plot_scatter(axes[1, 0], embedding_dk, test_indices, 'Test: Deep Kernel -> UMAP')
    plot_scatter(axes[1, 1], embedding_pca, test_indices, 'Test: BERT -> PCA -> UMAP')
    plot_scatter(axes[1, 2], embedding_direct, test_indices, 'Test: BERT -> UMAP')
    
    # Colorbar (shared)
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label='Accuracy')
    
    output_path = results_dir / "split_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Split visualization saved to {output_path}")

    # Create MLP projection visualization (using trained encoders from GP)
    visualize_mlp_projections(hbbops, all_results, id_to_idx, results_dir)


if __name__ == "__main__":
    main()
