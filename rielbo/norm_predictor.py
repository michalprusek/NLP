"""Norm Predictor for Decoupled Direction & Magnitude approach.

Maps normalized direction vectors to their original magnitudes.
This allows spherical flow to work on unit sphere while preserving
magnitude information needed by decoders.

Mathematical formulation:
- Embedding x ∈ R^d can be decomposed as x = r * u where:
  - u = x/||x|| ∈ S^{d-1} (direction on unit sphere)
  - r = ||x|| ∈ R^+ (magnitude)
- Spherical flow operates on u (direction)
- NormPredictor learns f: S^{d-1} → R^+ mapping direction to magnitude

Usage:
    # Training
    norm_predictor = NormPredictor(input_dim=256)
    norm_predictor.fit(embeddings, epochs=100)

    # Inference
    directions = F.normalize(embeddings, p=2, dim=-1)
    predicted_norms = norm_predictor(directions)
    reconstructed = directions * predicted_norms
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class NormPredictor(nn.Module):
    """MLP that predicts embedding magnitude from direction.

    Input: Normalized vector on unit sphere [B, D]
    Output: Scalar magnitude [B, 1]
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensure positive output
        )

    def forward(self, x_normalized: torch.Tensor) -> torch.Tensor:
        """Predict magnitude for normalized direction vectors.

        Args:
            x_normalized: Unit vectors [B, D] with ||x|| = 1

        Returns:
            Predicted magnitudes [B, 1]
        """
        return self.net(x_normalized)

    def fit(
        self,
        embeddings: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "cuda",
    ) -> dict:
        """Train norm predictor on embeddings.

        Args:
            embeddings: Raw embeddings [N, D]
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            device: Device for training

        Returns:
            Training statistics
        """
        self.to(device)
        self.train()

        # Compute directions and norms
        norms = embeddings.norm(dim=-1, keepdim=True)
        directions = embeddings / norms.clamp(min=1e-8)

        logger.info(f"Training NormPredictor on {len(embeddings)} samples")
        logger.info(f"  Norm stats: mean={norms.mean():.4f}, std={norms.std():.4f}")

        # Create dataset
        dataset = TensorDataset(directions, norms)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self.parameters(), lr=lr)

        best_loss = float('inf')
        history = []

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches = 0

            for batch_dir, batch_norm in loader:
                batch_dir = batch_dir.to(device)
                batch_norm = batch_norm.to(device)

                pred_norm = self(batch_dir)
                loss = F.mse_loss(pred_norm, batch_norm)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            history.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss

            if epoch % 20 == 0 or epoch == 1:
                logger.info(f"  Epoch {epoch}/{epochs}: loss={avg_loss:.6f}")

        self.eval()

        # Compute final metrics
        with torch.no_grad():
            all_dirs = directions.to(device)
            all_norms = norms.to(device)
            pred_norms = self(all_dirs)

            mae = (pred_norms - all_norms).abs().mean().item()
            mape = ((pred_norms - all_norms).abs() / all_norms.clamp(min=1e-8)).mean().item() * 100

        logger.info(f"  Final: MAE={mae:.4f}, MAPE={mape:.2f}%")

        return {
            "best_loss": best_loss,
            "mae": mae,
            "mape": mape,
            "history": history,
        }

    def save(self, path: str) -> None:
        """Save model to file."""
        torch.save({
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
        }, path)
        logger.info(f"Saved NormPredictor to {path}")

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "NormPredictor":
        """Load model from file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(input_dim=checkpoint["input_dim"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        logger.info(f"Loaded NormPredictor from {path}")
        return model


def train_norm_predictor_guacamol(
    output_path: str = "rielbo/checkpoints/guacamol_flow_spherical/norm_predictor.pt",
    n_samples: int = 10000,
    epochs: int = 100,
    device: str = "cuda",
    use_zinc: bool = False,
    zinc_path: str = "datasets/zinc/zinc_all.txt",
) -> NormPredictor:
    """Train norm predictor on SELFIES VAE embeddings.

    Args:
        output_path: Path to save trained model
        n_samples: Number of molecules to use
        epochs: Training epochs
        device: Device for training
        use_zinc: Use ZINC dataset instead of GuacaMol
        zinc_path: Path to ZINC SMILES file

    Returns:
        Trained NormPredictor
    """
    from shared.guacamol.codec import SELFIESVAECodec

    logger.info("Loading data and codec...")
    codec = SELFIESVAECodec.from_pretrained(device=device)

    if use_zinc:
        from shared.guacamol.data import load_zinc_smiles
        smiles_list = load_zinc_smiles(path=zinc_path, n_samples=n_samples)
    else:
        from shared.guacamol.data import load_guacamol_data
        smiles_list, _, _ = load_guacamol_data(n_samples=n_samples, task_id="pdop")

    # Encode molecules
    logger.info(f"Encoding {len(smiles_list)} molecules...")
    embeddings = []
    batch_size = 64
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Encoding"):
        batch = smiles_list[i:i + batch_size]
        with torch.no_grad():
            emb = codec.encode(batch)
        embeddings.append(emb.cpu())
    embeddings = torch.cat(embeddings, dim=0)

    # Train norm predictor
    input_dim = embeddings.shape[1]
    norm_predictor = NormPredictor(input_dim=input_dim)

    stats = norm_predictor.fit(
        embeddings,
        epochs=epochs,
        device=device,
    )

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    norm_predictor.save(output_path)

    return norm_predictor


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train NormPredictor")
    parser.add_argument("--output", type=str,
                       default="rielbo/checkpoints/guacamol_flow_spherical/norm_predictor.pt")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--zinc", action="store_true",
                       help="Use ZINC dataset instead of GuacaMol")
    parser.add_argument("--zinc-path", type=str, default="datasets/zinc/zinc_all.txt",
                       help="Path to ZINC SMILES file")

    args = parser.parse_args()

    train_norm_predictor_guacamol(
        output_path=args.output,
        n_samples=args.n_samples,
        epochs=args.epochs,
        device=args.device,
        use_zinc=args.zinc,
        zinc_path=args.zinc_path,
    )
