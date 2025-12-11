"""
Model Persistence for HbBoPs Improved 3

This module provides save/load functionality for trained GP models,
allowing:
1. Saving trained GP at the end of HbBoPs run
2. Loading pre-trained GP for inference on new prompts
3. Resuming optimization from a checkpoint
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import gpytorch

from .gp_model import (
    MultiFidelityDeepKernelGP,
    FeatureExtractor,
    GPNormalizationParams
)


class GPModelSaver:
    """
    Save and load trained GP models for inference.

    Checkpoint contents:
    - gp_model_state_dict: All GP parameters (kernel, mean)
    - likelihood_state_dict: Noise parameters
    - feature_extractor_state_dict: Deep kernel network weights
    - norm_params: Normalization parameters for input/output
    - model_config: Architecture configuration
    - design_data: Training data for potential resumption
    - metadata: Additional info (best error, num observations, etc.)
    """

    @staticmethod
    def save(
        filepath: Path,
        gp_model: MultiFidelityDeepKernelGP,
        likelihood: gpytorch.likelihoods.FixedNoiseGaussianLikelihood,
        feature_extractor: FeatureExtractor,
        norm_params: GPNormalizationParams,
        design_data: Optional[List[Tuple]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save complete GP state for later inference or resumption.

        Args:
            filepath: Path to save checkpoint (.pt file)
            gp_model: Trained MultiFidelityDeepKernelGP
            likelihood: FixedNoiseGaussianLikelihood
            feature_extractor: Trained FeatureExtractor
            norm_params: GPNormalizationParams with normalization info
            design_data: Optional training data for resumption
            metadata: Optional additional metadata
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'gp_model_state_dict': gp_model.state_dict(),
            'likelihood_state_dict': likelihood.state_dict(),
            'feature_extractor_state_dict': feature_extractor.state_dict(),
            'norm_params': norm_params.to_dict(),
            'model_config': {
                'input_dim': feature_extractor.input_dim,
                'latent_dim': feature_extractor.latent_dim,
                'num_fidelities': gp_model.num_fidelities
            },
            'design_data': design_data,
            'metadata': metadata or {}
        }

        torch.save(checkpoint, filepath)
        print(f"GP model saved to: {filepath}")

    @staticmethod
    def load(
        filepath: Path,
        device: torch.device = None
    ) -> Dict[str, Any]:
        """
        Load GP model for inference.

        The loaded model is ready for:
        - Making predictions (expected improvement)
        - Evaluating new prompts
        - Continuing optimization

        Args:
            filepath: Path to saved checkpoint
            device: Target device (defaults to CPU)

        Returns:
            Dictionary with:
            - gp_model: Loaded MultiFidelityDeepKernelGP
            - likelihood: Loaded FixedNoiseGaussianLikelihood
            - feature_extractor: Loaded FeatureExtractor
            - norm_params: Loaded GPNormalizationParams
            - design_data: Original training data (if saved)
            - metadata: Original metadata
        """
        device = device or torch.device('cpu')
        filepath = Path(filepath)

        checkpoint = torch.load(filepath, map_location=device, weights_only=False)

        # Reconstruct feature extractor
        config = checkpoint['model_config']
        feature_extractor = FeatureExtractor(
            input_dim=config['input_dim'],
            latent_dim=config['latent_dim']
        ).to(device)
        feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])

        # Reconstruct normalization params
        norm_params = GPNormalizationParams.from_dict(
            checkpoint['norm_params'],
            device=device
        )

        # Create dummy training data for GP initialization
        # Actual weights will be loaded from state_dict
        # Note: train_x now includes fidelity as last column
        dummy_x = torch.zeros(1, config['input_dim'] * 2 + 1, device=device)
        dummy_y = torch.zeros(1, device=device)
        dummy_noise = torch.ones(1, device=device) * 0.01

        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=dummy_noise,
            learn_additional_noise=False
        ).to(device)

        gp_model = MultiFidelityDeepKernelGP(
            train_x=dummy_x,
            train_y=dummy_y,
            likelihood=likelihood,
            feature_extractor=feature_extractor,
            num_fidelities=config['num_fidelities'],
            input_dim=config['input_dim']
        ).to(device)

        # Load actual weights
        gp_model.load_state_dict(checkpoint['gp_model_state_dict'])
        likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

        # Set to eval mode for inference
        gp_model.eval()
        likelihood.eval()

        print(f"GP model loaded from: {filepath}")
        if checkpoint.get('metadata'):
            meta = checkpoint['metadata']
            if 'num_observations' in meta:
                print(f"  Trained on {meta['num_observations']} observations")
            if 'best_error' in meta:
                print(f"  Best validation error: {meta['best_error']:.4f}")

        return {
            'gp_model': gp_model,
            'likelihood': likelihood,
            'feature_extractor': feature_extractor,
            'norm_params': norm_params,
            'design_data': checkpoint.get('design_data'),
            'metadata': checkpoint.get('metadata', {}),
            'model_config': config
        }

    @staticmethod
    def save_metadata_json(
        filepath: Path,
        norm_params: GPNormalizationParams,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Save human-readable metadata as JSON.

        Useful for quick inspection without loading the full model.
        """
        filepath = Path(filepath)

        json_data = {
            'fidelity_levels': list(norm_params.fidelity_to_idx.keys()),
            'fidelity_to_idx': {str(k): v for k, v in norm_params.fidelity_to_idx.items()},
            'y_mean_logit': norm_params.y_mean_logit,
            'y_std_logit': norm_params.y_std_logit,
            'epsilon': norm_params.epsilon,
            **metadata
        }

        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)


def update_gp_with_new_data(
    loaded: Dict[str, Any],
    new_observations: List[Tuple],
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Update a loaded GP with new observations.

    This is useful for continuing optimization from a checkpoint.
    Note: This retrains the GP from scratch with the combined data.

    Args:
        loaded: Dictionary from GPModelSaver.load()
        new_observations: List of (prompt_idx, inst_emb, ex_emb, val_error, fidelity)
        device: Target device

    Returns:
        Updated model dictionary (requires retraining)
    """
    device = device or torch.device('cpu')

    # Combine old and new design data
    old_data = loaded.get('design_data') or []
    combined_data = list(old_data) + list(new_observations)

    # Update metadata
    metadata = loaded.get('metadata', {}).copy()
    metadata['num_observations'] = len(combined_data)
    metadata['continued_from'] = metadata.get('num_observations', 0)

    return {
        **loaded,
        'design_data': combined_data,
        'metadata': metadata,
        'needs_retraining': True
    }
