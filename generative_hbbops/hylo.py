"""
HyLO: Hyperband Latent Optimization for prompt tuning.

Main orchestrator that:
1. Loads data and samples initial prompts
2. Trains GP on sampled prompts
3. Runs embedding optimization (Strategy A or B)
4. Inverts optimized embeddings via Vec2Text
5. Creates visualizations
6. Outputs final prompt with GP-predicted error rate

Usage:
    uv run python generative_hbbops/hylo.py \
        --strategy coordinate_descent \
        --n-samples 4 \
        --vec2text-steps 50
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

from .config import HyLOConfig
from .encoder import GTREncoder
from .gp_model import GPTrainer
from .optimizer import CoordinateDescentOptimizer, GumbelSoftmaxOptimizer, OptimizationResult
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


class HyLO:
    """Main HyLO optimization pipeline."""

    def __init__(self, config: HyLOConfig):
        """
        Args:
            config: HyLOConfig with all hyperparameters
        """
        self.config = config
        self.device = self._get_device(config.device)

        # Set random seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Initialize components (lazy loading)
        self._encoder: Optional[GTREncoder] = None
        self._gp_trainer: Optional[GPTrainer] = None
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
    def gp_trainer(self) -> GPTrainer:
        """Lazy-load GP trainer."""
        if self._gp_trainer is None:
            self._gp_trainer = GPTrainer(
                latent_dim=self.config.latent_dim,
                train_epochs=self.config.gp_train_epochs,
                lr=self.config.gp_lr,
                patience=self.config.gp_patience,
                device=self.device,
                use_leaky_relu=self.config.use_leaky_relu,
                leaky_relu_slope=self.config.leaky_relu_slope
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
        """Load prompts from full_grid_combined.jsonl.

        Expected format per line:
        {
            "prompt_idx": int,
            "instruction_id": int,
            "exemplar_id": int,
            "error_rate": float,
            "timestamp": str
        }
        """
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

        Returns:
            List of prompt dicts with embeddings and error rates
        """
        n = self.config.n_initial_samples

        if n == -1:
            # Use all prompts
            print(f"Using ALL {len(self.all_prompts)} prompts for GP training...")
            return self.all_prompts.copy()

        if select_best:
            # Select n best prompts (lowest error rate)
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
        """Pre-compute embeddings for all unique instructions/exemplars.

        Stores:
            self.instruction_embeddings: (num_instructions, 768)
            self.exemplar_embeddings: (num_exemplars, 768)
        """
        print("Encoding all instructions and exemplars...")

        # Encode instructions
        self.instruction_embeddings = self.encoder.encode_batch(self.instruction_texts)
        print(f"  Encoded {len(self.instruction_texts)} instructions")

        # Encode exemplars
        self.exemplar_embeddings = self.encoder.encode_batch(self.exemplar_texts)
        print(f"  Encoded {len(self.exemplar_texts)} exemplars")

    def train_gp(self, samples: List[Dict]) -> None:
        """Train GP on sampled prompts.

        Args:
            samples: List of prompt dicts
        """
        print(f"Training GP on {len(samples)} samples...")

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

        # Train
        self.gp_trainer.train(
            inst_embs, ex_embs, error_rates,
            verbose=True
        )

    def compute_ei_for_all(self, vmin_b: float) -> np.ndarray:
        """Compute EI for all prompts using trained GP.

        Args:
            vmin_b: Best observed error rate

        Returns:
            (N,) EI values for all prompts
        """
        print("Computing EI for all prompts...")

        # Build full embedding matrix
        N = len(self.all_prompts)
        inst_embs = np.zeros((N, 768))
        ex_embs = np.zeros((N, 768))

        for i, p in enumerate(self.all_prompts):
            inst_embs[i] = self.instruction_embeddings[p['instruction_id']]
            ex_embs[i] = self.exemplar_embeddings[p['exemplar_id']]

        # Convert to tensors
        inst_tensor = torch.tensor(inst_embs, dtype=torch.float32, device=self.device)
        ex_tensor = torch.tensor(ex_embs, dtype=torch.float32, device=self.device)

        # Compute EI batch
        ei_values = self.gp_trainer.compute_ei_batch(inst_tensor, ex_tensor, vmin_b)

        return ei_values

    def get_latent_features_for_all(self) -> np.ndarray:
        """Get latent features for all prompts.

        Returns:
            (N, latent_dim) latent features
        """
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
        vmin_b: float
    ) -> OptimizationResult:
        """Run embedding optimization using configured strategy.

        Args:
            best_sample: Best prompt from initial samples
            vmin_b: Best observed error rate

        Returns:
            OptimizationResult with final embeddings
        """
        print(f"\nRunning {self.config.strategy} optimization...")

        # Get initial embeddings
        init_inst_emb = torch.tensor(
            self.instruction_embeddings[best_sample['instruction_id']],
            dtype=torch.float32, device=self.device
        )
        init_ex_idx = best_sample['exemplar_id']

        if self.config.strategy == "coordinate_descent":
            optimizer = CoordinateDescentOptimizer(
                gp_trainer=self.gp_trainer,
                exemplar_embeddings=self.exemplar_embeddings,
                instruction_embeddings=self.instruction_embeddings,  # For multi-start
                n_steps=self.config.cd_n_steps,
                lr=self.config.cd_lr,
                convergence_threshold=self.config.cd_convergence_threshold,
                max_iterations=self.config.cd_max_iterations,
                n_restarts=self.config.cd_n_restarts,
                perturbation_scale=self.config.perturbation_scale,
                device=self.device,
                gradient_clip_norm=self.config.gradient_clip_norm,
                use_log_ei=self.config.use_log_ei,
                ei_epsilon=self.config.ei_epsilon
            )
            result = optimizer.optimize(init_inst_emb, init_ex_idx, vmin_b, verbose=True)

        else:  # gumbel_softmax
            optimizer = GumbelSoftmaxOptimizer(
                gp_trainer=self.gp_trainer,
                exemplar_embeddings=self.exemplar_embeddings,
                n_steps=self.config.gs_n_steps,
                lr=self.config.gs_lr,
                initial_temperature=self.config.gs_initial_temperature,
                final_temperature=self.config.gs_final_temperature,
                anneal_rate=self.config.gs_anneal_rate,
                device=self.device,
                gradient_clip_norm=self.config.gradient_clip_norm,
                use_log_ei=self.config.use_log_ei,
                ei_epsilon=self.config.ei_epsilon
            )
            result = optimizer.optimize(
                init_inst_emb, vmin_b,
                init_exemplar_idx=init_ex_idx,  # Pass initial exemplar for correct start
                verbose=True
            )

        return result

    def invert_embeddings(
        self,
        result: OptimizationResult
    ) -> Tuple[str, str]:
        """Convert optimized embeddings to text via Vec2Text.

        Args:
            result: OptimizationResult from optimizer

        Returns:
            (instruction_text, exemplar_text)
            Note: Exemplar text comes from selected exemplar, not inversion
        """
        print("\nInverting optimized embeddings via Vec2Text...")

        try:
            inverter = Vec2TextInverter(
                num_steps=self.config.vec2text_num_steps,
                beam_width=self.config.vec2text_beam_width,
                device=str(self.device)
            )

            instruction_text = inverter.invert_instruction(
                result.optimized_instruction_emb,
                verbose=True
            )
        except Exception as e:
            print(f"Vec2Text inversion failed: {e}")
            print("Falling back to nearest neighbor...")

            # Fallback to nearest neighbor
            nn_inverter = NearestNeighborInverter(
                texts=self.instruction_texts,
                embeddings=torch.tensor(self.instruction_embeddings, device=self.device),
                device=self.device
            )
            instruction_text, nn_idx, nn_dist = nn_inverter.invert(
                result.optimized_instruction_emb
            )
            print(f"Nearest instruction (id={nn_idx}, dist={nn_dist:.4f}): {instruction_text}")

        # Exemplar comes from selection, not inversion
        exemplar_text = self.exemplar_texts[result.selected_exemplar_idx]

        return instruction_text, exemplar_text

    def verify_inversion(
        self,
        instruction_text: str,
        optimized_emb: torch.Tensor
    ) -> Tuple[float, torch.Tensor]:
        """Re-encode inverted text and compute similarity.

        Args:
            instruction_text: Inverted instruction text
            optimized_emb: Original optimized embedding

        Returns:
            (cosine_similarity, reembedded_vector)
        """
        print("\nVerifying inversion quality...")

        # Re-encode
        reembedded = self.encoder.encode_tensor(instruction_text)

        # Compute similarity
        cosine_sim = self.encoder.compute_cosine_similarity(optimized_emb, reembedded)

        print(f"Cosine similarity: {cosine_sim:.4f}")

        return cosine_sim, reembedded

    def predict_error_rate(
        self,
        instruction_emb: torch.Tensor,
        exemplar_emb: torch.Tensor
    ) -> Tuple[float, float]:
        """Use GP to predict error rate for final prompt.

        Args:
            instruction_emb: Instruction embedding
            exemplar_emb: Exemplar embedding

        Returns:
            (predicted_mean, predicted_std)
        """
        mean, std = self.gp_trainer.predict(instruction_emb, exemplar_emb)
        return float(mean.item()), float(std.item())

    def run(self, select_best: bool = False) -> Dict:
        """Execute full HyLO pipeline.

        Args:
            select_best: If True, select N best prompts instead of random

        Steps:
            1. Load data
            2. Sample initial prompts
            3. Encode all texts
            4. Train GP
            5. Create visualization 1
            6. Run optimization
            7. Create visualization 2
            8. Invert embeddings
            9. Verify inversion
            10. Create visualization 3
            11. Predict error rate
            12. Save results

        Returns:
            Results dict with all outputs
        """
        print("=" * 60)
        print("HyLO: Hyperband Latent Optimization")
        print("=" * 60)
        print(f"Strategy: {self.config.strategy}")
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

        # 4. Train GP
        self.train_gp(samples)

        # Get EI and latent features for visualization
        ei_values = self.compute_ei_for_all(vmin_b)
        latent_features = self.get_latent_features_for_all()
        error_rates = np.array([p['error_rate'] for p in self.all_prompts])

        # Compute training indices for visualization
        # Map sample prompt_idx to index in all_prompts
        prompt_idx_to_idx = {p['prompt_idx']: i for i, p in enumerate(self.all_prompts)}
        training_indices = np.array([prompt_idx_to_idx[s['prompt_idx']] for s in samples])

        visualization_paths = []

        # 5. Visualization 1
        if self.config.save_visualizations:
            print("\n" + "=" * 60)
            print("Creating Visualization 1: Initial GP")
            path1 = self.visualizer.plot_initial_gp(
                latent_features, error_rates, ei_values, vmin_b,
                training_indices=training_indices
            )
            visualization_paths.append(str(path1))

        # 6. Run optimization
        print("\n" + "=" * 60)
        result = self.run_optimization(best_sample, vmin_b)

        # 7. Visualization 2
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
            result.optimized_instruction_emb
        )

        # 10. Visualization 3
        if self.config.save_visualizations:
            print("\n" + "=" * 60)
            print("Creating Visualization 3: Inversion Verification")

            # Get latent features for optimized and re-embedded
            opt_latent = self.gp_trainer.get_latent_features(
                result.optimized_instruction_emb,
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

        # 11. Predict error rate
        print("\n" + "=" * 60)
        print("GP Prediction for final prompt...")

        # Use re-embedded instruction for prediction (more realistic)
        pred_mean, pred_std = self.predict_error_rate(
            reembedded,
            result.selected_exemplar_emb
        )
        print(f"Predicted error rate: {pred_mean:.4f} +/- {pred_std:.4f}")
        print(f"Predicted accuracy: {1-pred_mean:.4f}")

        # 12. Compile results
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
                'selected_exemplar_idx': result.selected_exemplar_idx
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
            'visualizations': visualization_paths
        }

        # Save results
        output_path = self.save_results(results)
        results['output_path'] = str(output_path)

        return results

    def save_results(self, results: Dict) -> Path:
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hylo_{timestamp}.json"
        output_path = Path(self.config.output_dir) / filename

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to {output_path}")
        return output_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='HyLO: Hyperband Latent Optimization for prompt tuning'
    )
    parser.add_argument(
        '--strategy',
        choices=['coordinate_descent', 'gumbel_softmax'],
        default='coordinate_descent',
        help='Optimization strategy'
    )
    parser.add_argument(
        '--n-samples', type=int, default=4,
        help='Number of initial samples for GP training'
    )
    parser.add_argument(
        '--use-all-samples', action='store_true',
        help='Use all 625 prompts for GP training (overrides --n-samples)'
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
        '--cd-steps', type=int, default=500,
        help='Gradient steps per coordinate descent iteration'
    )
    parser.add_argument(
        '--cd-lr', type=float, default=0.01,
        help='Learning rate for coordinate descent'
    )
    parser.add_argument(
        '--gs-steps', type=int, default=1000,
        help='Total steps for Gumbel-Softmax optimization'
    )
    parser.add_argument(
        '--gs-lr', type=float, default=0.01,
        help='Learning rate for Gumbel-Softmax'
    )
    parser.add_argument(
        '--gs-initial-temp', type=float, default=5.0,
        help='Initial temperature for Gumbel-Softmax'
    )
    parser.add_argument(
        '--gs-final-temp', type=float, default=0.1,
        help='Final temperature for Gumbel-Softmax'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='/home/prusek/NLP/generative_hbbops/results',
        help='Output directory'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--no-visualize', action='store_true',
        help='Skip visualization generation'
    )
    parser.add_argument(
        '--select-best', action='store_true',
        help='Select N best prompts instead of random'
    )

    # Gradient stability arguments
    parser.add_argument(
        '--use-log-ei', action='store_true',
        help='Use log(EI) transformation for better gradient flow'
    )
    parser.add_argument(
        '--gradient-clip', type=float, default=None,
        help='Gradient clipping norm (None = disabled)'
    )
    parser.add_argument(
        '--perturbation-scale', type=float, default=0.1,
        help='Basin hopping perturbation scale'
    )
    parser.add_argument(
        '--ei-epsilon', type=float, default=1e-8,
        help='Epsilon for numerical stability in EI computation'
    )

    # Feature extractor arguments
    parser.add_argument(
        '--use-leaky-relu', action='store_true',
        help='Use LeakyReLU instead of ReLU in feature extractor'
    )
    parser.add_argument(
        '--leaky-relu-slope', type=float, default=0.01,
        help='Negative slope for LeakyReLU'
    )

    # Multi-start arguments
    parser.add_argument(
        '--n-restarts', type=int, default=5,
        help='Number of random restarts for coordinate descent'
    )

    args = parser.parse_args()

    # Use -1 for all samples
    n_samples = -1 if args.use_all_samples else args.n_samples

    config = HyLOConfig(
        strategy=args.strategy,
        n_initial_samples=n_samples,
        vec2text_num_steps=args.vec2text_steps,
        vec2text_beam_width=args.vec2text_beam_width,
        cd_n_steps=args.cd_steps,
        cd_lr=args.cd_lr,
        gs_n_steps=args.gs_steps,
        gs_lr=args.gs_lr,
        gs_initial_temperature=args.gs_initial_temp,
        gs_final_temperature=args.gs_final_temp,
        output_dir=args.output_dir,
        seed=args.seed,
        save_visualizations=not args.no_visualize,
        # Gradient stability
        use_log_ei=args.use_log_ei,
        gradient_clip_norm=args.gradient_clip,
        ei_epsilon=args.ei_epsilon,
        perturbation_scale=args.perturbation_scale,
        cd_n_restarts=args.n_restarts,
        # Feature extractor
        use_leaky_relu=args.use_leaky_relu,
        leaky_relu_slope=args.leaky_relu_slope
    )

    hylo = HyLO(config)
    results = hylo.run(select_best=args.select_best)

    print("\n" + "=" * 60)
    print("HyLO Optimization Complete")
    print("=" * 60)
    print(f"Strategy: {config.strategy}")
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
