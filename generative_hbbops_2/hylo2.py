"""
HyLO2: Latent Space Optimization for prompt tuning.

Extension of HyLO with:
1. LatentProjector (10D -> 768D) trained jointly with GP
2. Optimization directly in 10D latent space instead of 768D
3. Improved Vec2Text alignment through reconstruction training

Usage:
    uv run python generative_hbbops_2/hylo2.py --n-samples 4 --vec2text-steps 50
"""
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np

from .config import HyLO2Config
from .encoder import GTREncoder
from .gp_model import GPTrainer2
from .optimizer import LatentSpaceOptimizer, LatentOptimizationResult
from .inverter import Vec2TextInverter, NearestNeighborInverter
from .visualizer import HyLOVisualizer


def load_instructions(path: str) -> List[str]:
    """Load instructions from file (one per line, with index prefix)."""
    instructions = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Remove index prefix if present (e.g., "1. instruction" -> "instruction")
            if line[0].isdigit() and '. ' in line:
                line = line.split('. ', 1)[1]
            instructions.append(line)
    return instructions


def load_exemplars(path: str) -> List[str]:
    """Load exemplars from file (separated by ===... lines)."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by separator
    separator = '=' * 80
    blocks = content.split(separator)

    exemplars = []
    for block in blocks:
        block = block.strip()
        if block:
            exemplars.append(block)

    return exemplars


class HyLO2:
    """Main HyLO2 optimization pipeline with latent space optimization."""

    def __init__(self, config: HyLO2Config):
        """
        Args:
            config: HyLO2Config with all hyperparameters
        """
        self.config = config
        self.device = self._get_device(config.device)

        # Set random seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Initialize components (lazy loading)
        self._encoder: Optional[GTREncoder] = None
        self._gp_trainer: Optional[GPTrainer2] = None
        self._visualizer: Optional[HyLOVisualizer] = None

        # Data storage
        self.all_prompts: List[Dict] = []
        self.instruction_texts: List[str] = []
        self.exemplar_texts: List[str] = []
        self.instruction_embeddings: Optional[np.ndarray] = None
        self.exemplar_embeddings: Optional[np.ndarray] = None

    def _get_device(self, device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    @property
    def encoder(self) -> GTREncoder:
        """Lazy-load encoder."""
        if self._encoder is None:
            self._encoder = GTREncoder(
                self.config.encoder_name,
                device=str(self.device)
            )
        return self._encoder

    @property
    def gp_trainer(self) -> GPTrainer2:
        """Lazy-load GP trainer with LatentProjector."""
        if self._gp_trainer is None:
            self._gp_trainer = GPTrainer2(
                latent_dim=self.config.latent_dim,
                train_epochs=self.config.gp_train_epochs,
                lr=self.config.gp_lr,
                patience=self.config.gp_patience,
                device=self.device,
                use_leaky_relu=self.config.use_leaky_relu,
                leaky_relu_slope=self.config.leaky_relu_slope,
                reconstruction_weight=self.config.reconstruction_weight,
                warmup_epochs=self.config.warmup_epochs
            )
        return self._gp_trainer

    @property
    def visualizer(self) -> HyLOVisualizer:
        """Lazy-load visualizer."""
        if self._visualizer is None:
            self._visualizer = HyLOVisualizer(
                output_dir=self.config.output_dir,
                dpi=self.config.visualization_dpi
            )
        return self._visualizer

    def load_data(self) -> None:
        """Load prompts from full_grid_combined.jsonl."""
        print(f"Loading data from {self.config.data_path}...")

        # Load prompts
        with open(self.config.data_path, 'r') as f:
            for line in f:
                self.all_prompts.append(json.loads(line))

        print(f"Loaded {len(self.all_prompts)} prompts")

        # Load instruction and exemplar texts
        self.instruction_texts = load_instructions(self.config.instructions_path)
        self.exemplar_texts = load_exemplars(self.config.exemplars_path)

        print(f"Loaded {len(self.instruction_texts)} instructions, "
              f"{len(self.exemplar_texts)} exemplars")

    def sample_initial_prompts(self, select_best: bool = False) -> List[Dict]:
        """Sample prompts for GP training.

        If n_initial_samples == -1, use ALL prompts.
        If select_best == True, select n prompts with lowest error rate.
        """
        n = self.config.n_initial_samples

        if n == -1:
            print(f"Using ALL {len(self.all_prompts)} prompts for GP training...")
            return self.all_prompts.copy()

        if select_best:
            sorted_prompts = sorted(self.all_prompts, key=lambda p: p['error_rate'])
            sampled = sorted_prompts[:n]
            print(f"Selecting {n} BEST prompts for GP training...")
            for i, p in enumerate(sampled):
                print(f"  {i+1}. inst={p['instruction_id']}, ex={p['exemplar_id']}, "
                      f"error={p['error_rate']:.4f}, acc={1-p['error_rate']:.2%}")
        else:
            print(f"Sampling {n} random prompts for GP training...")
            sampled = random.sample(self.all_prompts, n)

        return sampled

    def encode_all_texts(self) -> None:
        """Pre-compute embeddings for all unique instructions/exemplars."""
        print("Encoding all instructions and exemplars...")

        self.instruction_embeddings = self.encoder.encode_batch(self.instruction_texts)
        print(f"  Encoded {len(self.instruction_texts)} instructions")

        self.exemplar_embeddings = self.encoder.encode_batch(self.exemplar_texts)
        print(f"  Encoded {len(self.exemplar_texts)} exemplars")

    def train_gp(self, samples: List[Dict]) -> None:
        """Train GP + FeatureExtractor + LatentProjector jointly."""
        print(f"Training GP with LatentProjector on {len(samples)} samples...")

        # Prepare data
        inst_embs = []
        ex_embs = []
        error_rates = []

        for s in samples:
            inst_id = s['instruction_id']
            ex_id = s['exemplar_id']

            inst_embs.append(self.instruction_embeddings[inst_id])
            ex_embs.append(self.exemplar_embeddings[ex_id])
            error_rates.append(s['error_rate'])

        inst_embs = np.array(inst_embs)
        ex_embs = np.array(ex_embs)
        error_rates = np.array(error_rates)

        # Train with joint reconstruction loss
        self.gp_trainer.train(
            inst_embs, ex_embs, error_rates,
            verbose=True
        )

    def compute_ei_for_all(self, vmin_b: float) -> np.ndarray:
        """Compute EI for all prompts using trained GP."""
        print("Computing EI for all prompts...")

        N = len(self.all_prompts)
        ei_values = np.zeros(N)

        for i, p in enumerate(self.all_prompts):
            inst_emb = torch.tensor(
                self.instruction_embeddings[p['instruction_id']],
                dtype=torch.float32, device=self.device
            )
            ex_emb = torch.tensor(
                self.exemplar_embeddings[p['exemplar_id']],
                dtype=torch.float32, device=self.device
            )

            # Get latent and compute EI
            with torch.no_grad():
                latent = self.gp_trainer.get_latent_features(inst_emb, ex_emb)
                ei = self.gp_trainer.compute_ei_in_latent_space(latent, vmin_b)
                ei_values[i] = ei.item()

        return ei_values

    def get_latent_features_for_all(self) -> np.ndarray:
        """Get latent features for all prompts."""
        N = len(self.all_prompts)
        latents = []

        for p in self.all_prompts:
            inst_emb = torch.tensor(
                self.instruction_embeddings[p['instruction_id']],
                dtype=torch.float32, device=self.device
            )
            ex_emb = torch.tensor(
                self.exemplar_embeddings[p['exemplar_id']],
                dtype=torch.float32, device=self.device
            )

            with torch.no_grad():
                latent = self.gp_trainer.get_latent_features(inst_emb, ex_emb)
                latents.append(latent.cpu().numpy())

        return np.array(latents)

    def run_optimization(
        self,
        best_sample: Dict,
        vmin_b: float,
        top_k_samples: List[Dict] = None
    ) -> LatentOptimizationResult:
        """Run latent space optimization with multi-start from top K samples.

        Key difference from HyLO: Optimizes directly in 10D latent space
        instead of 768D embedding space.

        Args:
            best_sample: Best sample for fallback
            vmin_b: Best observed error rate
            top_k_samples: Top K samples to use as starting points
        """
        print("\nRunning Latent Space Optimization...")

        # Get initial embeddings
        init_inst_emb = torch.tensor(
            self.instruction_embeddings[best_sample['instruction_id']],
            dtype=torch.float32, device=self.device
        )
        init_ex_idx = best_sample['exemplar_id']

        optimizer = LatentSpaceOptimizer(
            gp_trainer=self.gp_trainer,
            exemplar_embeddings=self.exemplar_embeddings,
            n_steps=self.config.latent_n_steps,
            lr=self.config.latent_lr,
            convergence_threshold=self.config.latent_convergence_threshold,
            max_iterations=self.config.latent_max_iterations,
            n_restarts=self.config.latent_n_restarts,
            perturbation_scale=self.config.perturbation_scale,
            device=self.device,
            use_log_ei=self.config.use_log_ei,
            ei_epsilon=self.config.ei_epsilon,
            latent_bounds_sigma=self.config.latent_bounds_sigma,
            use_latent_bounds=self.config.use_latent_bounds
        )

        result = optimizer.optimize(
            init_inst_emb, init_ex_idx, vmin_b,
            top_k_samples=top_k_samples,
            instruction_embeddings=self.instruction_embeddings,
            verbose=True
        )
        return result

    def invert_embeddings(
        self,
        result: LatentOptimizationResult
    ) -> Tuple[str, str]:
        """Convert optimized embeddings to text via Vec2Text.

        The projected_instruction_emb comes from LatentProjector,
        which was trained to produce embeddings in Vec2Text's distribution.
        """
        print("\nInverting optimized embeddings via Vec2Text...")
        print(f"Using embedding from LatentProjector (trained with reconstruction loss)")

        try:
            inverter = Vec2TextInverter(
                num_steps=self.config.vec2text_num_steps,
                beam_width=self.config.vec2text_beam_width,
                device=str(self.device)
            )

            instruction_text = inverter.invert_instruction(
                result.projected_instruction_emb,
                verbose=True
            )
        except Exception as e:
            print(f"Vec2Text inversion failed: {e}")
            print("Falling back to nearest neighbor...")

            nn_inverter = NearestNeighborInverter(
                texts=self.instruction_texts,
                embeddings=torch.tensor(self.instruction_embeddings, device=self.device),
                device=self.device
            )
            instruction_text, nn_idx, nn_dist = nn_inverter.invert(
                result.projected_instruction_emb
            )
            print(f"Nearest instruction (id={nn_idx}, dist={nn_dist:.4f}): {instruction_text}")

        # Exemplar comes from selection, not inversion
        exemplar_text = self.exemplar_texts[result.selected_exemplar_idx]

        return instruction_text, exemplar_text

    def verify_inversion(
        self,
        instruction_text: str,
        projected_emb: torch.Tensor
    ) -> Tuple[float, torch.Tensor]:
        """Re-encode inverted text and compute similarity."""
        print("\nVerifying inversion quality...")

        # Re-encode
        reembedded = self.encoder.encode_tensor(instruction_text)

        # Compute similarity
        cosine_sim = self.encoder.compute_cosine_similarity(projected_emb, reembedded)

        print(f"Cosine similarity: {cosine_sim:.4f}")

        return cosine_sim, reembedded

    def predict_error_rate(
        self,
        instruction_emb: torch.Tensor,
        exemplar_emb: torch.Tensor
    ) -> Tuple[float, float]:
        """Use GP to predict error rate for final prompt."""
        mean, std = self.gp_trainer.predict(instruction_emb, exemplar_emb)
        return float(mean.item()), float(std.item())

    def run(self, select_best: bool = False) -> Dict:
        """Execute full HyLO2 pipeline.

        Steps:
            1. Load data
            2. Sample initial prompts
            3. Encode all texts
            4. Train GP with LatentProjector (joint training)
            5. Run latent space optimization
            6. Invert embeddings
            7. Verify inversion
            8. Predict error rate
            9. Save results
        """
        print("=" * 60)
        print("HyLO2: Latent Space Optimization")
        print("=" * 60)
        print(f"Latent dim: {self.config.latent_dim}")
        print(f"Reconstruction weight: {self.config.reconstruction_weight}")
        print(f"Warmup epochs: {self.config.warmup_epochs}")
        n_samples_str = "ALL" if self.config.n_initial_samples == -1 else str(self.config.n_initial_samples)
        selection_str = " (BEST)" if select_best else " (random)"
        print(f"Training samples: {n_samples_str}{selection_str}")
        print(f"Device: {self.device}")
        print()

        # 1. Load data
        self.load_data()

        # 2. Sample initial prompts
        samples = self.sample_initial_prompts(select_best=select_best)
        vmin_b = min(s['error_rate'] for s in samples)
        best_sample = min(samples, key=lambda s: s['error_rate'])
        print(f"Best initial sample: inst={best_sample['instruction_id']}, "
              f"ex={best_sample['exemplar_id']}, error={best_sample['error_rate']:.4f}")

        # 3. Encode all texts
        self.encode_all_texts()

        # 4. Train GP with LatentProjector
        self.train_gp(samples)

        # Get EI and latent features
        ei_values = self.compute_ei_for_all(vmin_b)
        latent_features = self.get_latent_features_for_all()
        error_rates = np.array([p['error_rate'] for p in self.all_prompts])

        # Compute training indices for visualization
        prompt_idx_to_idx = {p['prompt_idx']: i for i, p in enumerate(self.all_prompts)}
        training_indices = np.array([prompt_idx_to_idx[s['prompt_idx']] for s in samples])

        print(f"\nEI statistics: min={ei_values.min():.6f}, max={ei_values.max():.6f}, "
              f"mean={ei_values.mean():.6f}")

        visualization_paths = []

        # 5. Visualization 1: Initial GP
        if self.config.save_visualizations:
            print("\n" + "=" * 60)
            print("Creating Visualization 1: Initial GP")
            path1 = self.visualizer.plot_initial_gp(
                latent_features, error_rates, ei_values, vmin_b,
                training_indices=training_indices
            )
            visualization_paths.append(str(path1))

        # 6. Run latent space optimization (multi-start from top 5)
        print("\n" + "=" * 60)
        top_5_samples = sorted(samples, key=lambda s: s['error_rate'])[:5]
        result = self.run_optimization(best_sample, vmin_b, top_k_samples=top_5_samples)

        # 7. Visualization 2: Optimization trajectory
        if self.config.save_visualizations and result.trajectory:
            print("\n" + "=" * 60)
            print("Creating Visualization 2: Optimization Trajectory")
            path2 = self.visualizer.plot_optimization_trajectory(
                latent_features, error_rates, ei_values,
                result.trajectory, self.gp_trainer, vmin_b,
                training_indices=training_indices
            )
            visualization_paths.append(str(path2))

        # 8. Invert embeddings
        print("\n" + "=" * 60)
        instruction_text, exemplar_text = self.invert_embeddings(result)

        # 9. Verify inversion
        cosine_sim, reembedded = self.verify_inversion(
            instruction_text,
            result.projected_instruction_emb
        )

        # 10. Visualization 3: Inversion verification
        if self.config.save_visualizations:
            print("\n" + "=" * 60)
            print("Creating Visualization 3: Inversion Verification")

            # Get latent features for optimized and re-embedded
            opt_latent = self.gp_trainer.get_latent_features(
                result.projected_instruction_emb,
                result.selected_exemplar_emb
            ).cpu().numpy()

            reemb_latent = self.gp_trainer.get_latent_features(
                reembedded,
                result.selected_exemplar_emb
            ).cpu().numpy()

            path3 = self.visualizer.plot_inversion_verification(
                latent_features, error_rates, ei_values,
                opt_latent, reemb_latent, vmin_b, cosine_sim,
                training_indices=training_indices
            )
            visualization_paths.append(str(path3))

        # 8. Predict error rate
        print("\n" + "=" * 60)
        print("GP Prediction for final prompt...")

        pred_mean, pred_std = self.predict_error_rate(
            reembedded,
            result.selected_exemplar_emb
        )
        print(f"Predicted error rate: {pred_mean:.4f} +/- {pred_std:.4f}")
        print(f"Predicted accuracy: {1-pred_mean:.4f}")

        # 9. Compile results
        results = {
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat(),
            'initial_samples': samples,
            'vmin_b': vmin_b,
            'best_initial_sample': best_sample,
            'optimization': {
                'strategy': result.strategy,
                'n_iterations': result.n_iterations,
                'final_ei': result.final_ei,
                'selected_exemplar_idx': result.selected_exemplar_idx,
                'optimized_latent': result.optimized_latent.cpu().numpy().tolist()
            },
            'inversion': {
                'instruction_text': instruction_text,
                'exemplar_id': result.selected_exemplar_idx,
                'exemplar_text': exemplar_text[:500],  # Truncate for JSON
                'cosine_similarity': cosine_sim
            },
            'prediction': {
                'error_rate_mean': pred_mean,
                'error_rate_std': pred_std,
                'accuracy_mean': 1 - pred_mean
            },
            'latent_stats': {
                'mean': self.gp_trainer.gp_params.latent_mean.cpu().numpy().tolist(),
                'std': self.gp_trainer.gp_params.latent_std.cpu().numpy().tolist()
            },
            'visualizations': visualization_paths
        }

        # Save results
        output_path = self.save_results(results)
        results['output_path'] = str(output_path)

        return results

    def save_results(self, results: Dict) -> Path:
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hylo2_{timestamp}.json"
        output_path = Path(self.config.output_dir) / filename

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to {output_path}")
        return output_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='HyLO2: Latent Space Optimization for prompt tuning'
    )
    parser.add_argument(
        '--n-samples', type=int, default=4,
        help='Number of initial samples for GP training'
    )
    parser.add_argument(
        '--use-all-samples', action='store_true',
        help='Use all 625 prompts for GP training'
    )
    parser.add_argument(
        '--vec2text-steps', type=int, default=50,
        help='Number of Vec2Text correction steps'
    )
    parser.add_argument(
        '--vec2text-beam-width', type=int, default=4,
        help='Vec2Text beam search width'
    )
    parser.add_argument(
        '--latent-steps', type=int, default=500,
        help='Gradient steps for latent optimization'
    )
    parser.add_argument(
        '--latent-lr', type=float, default=0.1,
        help='Learning rate for latent optimization'
    )
    parser.add_argument(
        '--reconstruction-weight', type=float, default=1.0,
        help='Lambda for reconstruction loss (1.0 = equal weight with GP loss)'
    )
    parser.add_argument(
        '--warmup-epochs', type=int, default=500,
        help='GP-only epochs before adding reconstruction'
    )
    parser.add_argument(
        '--latent-bounds-sigma', type=float, default=3.0,
        help='Bound optimization to +/- N sigma'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='/home/prusek/NLP/generative_hbbops_2/results',
        help='Output directory'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--select-best', action='store_true',
        help='Select N best prompts instead of random'
    )
    parser.add_argument(
        '--use-log-ei', action='store_true',
        help='Use log(EI) transformation'
    )
    parser.add_argument(
        '--perturbation-scale', type=float, default=0.1,
        help='Perturbation scale for restarts'
    )
    parser.add_argument(
        '--n-restarts', type=int, default=5,
        help='Number of random restarts'
    )
    parser.add_argument(
        '--no-visualize', action='store_true',
        help='Skip visualization generation'
    )
    parser.add_argument(
        '--no-bounds', action='store_true',
        help='Disable latent space bounds (allow extrapolation)'
    )

    args = parser.parse_args()

    n_samples = -1 if args.use_all_samples else args.n_samples

    config = HyLO2Config(
        n_initial_samples=n_samples,
        vec2text_num_steps=args.vec2text_steps,
        vec2text_beam_width=args.vec2text_beam_width,
        latent_n_steps=args.latent_steps,
        latent_lr=args.latent_lr,
        reconstruction_weight=args.reconstruction_weight,
        warmup_epochs=args.warmup_epochs,
        latent_bounds_sigma=args.latent_bounds_sigma,
        latent_n_restarts=args.n_restarts,
        output_dir=args.output_dir,
        seed=args.seed,
        use_log_ei=args.use_log_ei,
        perturbation_scale=args.perturbation_scale,
        save_visualizations=not args.no_visualize,
        use_latent_bounds=not args.no_bounds
    )

    hylo = HyLO2(config)
    results = hylo.run(select_best=args.select_best)

    print("\n" + "=" * 60)
    print("HyLO2 Optimization Complete")
    print("=" * 60)
    print(f"Strategy: latent_space (10D)")
    n_samples_str = "ALL (625)" if config.n_initial_samples == -1 else str(config.n_initial_samples)
    print(f"Training samples: {n_samples_str}")
    print(f"Final EI: {results['optimization']['final_ei']:.6f}")
    print(f"Predicted error rate: {results['prediction']['error_rate_mean']:.4f} "
          f"+/- {results['prediction']['error_rate_std']:.4f}")
    print(f"\nGenerated instruction:")
    print(f"  {results['inversion']['instruction_text']}")
    print(f"\nSelected exemplar (id={results['inversion']['exemplar_id']}):")
    print(f"  {results['inversion']['exemplar_text'][:200]}...")
    print(f"\nInversion quality (cosine sim): {results['inversion']['cosine_similarity']:.4f}")
    print(f"\nResults saved to: {results['output_path']}")


if __name__ == "__main__":
    main()
