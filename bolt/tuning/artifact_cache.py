"""
Artifact Caching for BOLT Hyperparameter Tuning

Features:
- Hash-based caching of trained VAE, Scorer, and GP models
- Deterministic hashing of only relevant parameters
- Cache hit detection for skipping redundant training
- Metrics preservation for fast retrieval

Usage:
    cache = ArtifactCache(cache_dir)

    # Check if VAE exists
    if cache.has_vae(config):
        state_dict, metrics = cache.load_vae(config)
    else:
        # Train VAE
        cache.save_vae(config, vae.state_dict(), metrics)
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# Parameters that affect each artifact type
# If any of these change, the model must be retrained

VAE_AFFECTING_PARAMS = sorted([
    # VAE architecture
    "instruction_latent_dim",
    "exemplar_latent_dim",
    "set_transformer_hidden",
    "set_transformer_heads",
    "num_inducing",
    # VAE training
    "vae_beta",
    "vae_mse_weight",
    "vae_lr",
    "vae_epochs",
    "vae_annealing_epochs",
    "vae_patience",
    "vae_batch_size",
    # Scorer (affects VAE training in BOLT)
    "selection_weight",
    "scorer_hidden_dim",
    "cross_attn_heads",
    "mmr_lambda",
])

# GP depends on VAE + its own params
GP_AFFECTING_PARAMS = sorted(VAE_AFFECTING_PARAMS + [
    "gp_epochs",
    "gp_lr",
    "gp_patience",
    "use_deep_kernel",
    "dkl_output_dim",
    "dkl_hidden_dim",
    "use_product_kernel",
])

# Inference only affects final optimization, not model training
INFERENCE_ONLY_PARAMS = sorted([
    "ucb_beta",
    "ucb_beta_final",
    "num_restarts",
    "raw_samples",
    "distance_weight",
    "cosine_sim_threshold",
])


def compute_config_hash(
    config: Dict[str, Any],
    param_names: List[str],
) -> str:
    """
    Compute deterministic hash of config parameters.

    Args:
        config: Full hyperparameter config
        param_names: List of parameter names to include in hash

    Returns:
        SHA-256 hash string (64 chars)
    """
    # Extract only relevant params
    relevant = {}
    for name in param_names:
        if name in config:
            value = config[name]
            # Convert tensors/arrays to lists for JSON serialization
            if hasattr(value, 'tolist'):
                value = value.tolist()
            elif hasattr(value, 'item'):
                value = value.item()
            relevant[name] = value

    # Sort for determinism
    config_str = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()


@dataclass
class CacheEntry:
    """Metadata for a cached artifact."""
    hash_key: str
    artifact_type: str  # "vae", "scorer", "gp"
    path: str
    metrics: Dict[str, float]
    config_subset: Dict[str, Any]
    created_at: str
    accessed_count: int = 0
    last_accessed: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hash_key": self.hash_key,
            "artifact_type": self.artifact_type,
            "path": self.path,
            "metrics": self.metrics,
            "config_subset": self.config_subset,
            "created_at": self.created_at,
            "accessed_count": self.accessed_count,
            "last_accessed": self.last_accessed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CacheEntry:
        return cls(**d)


class ArtifactCache:
    """
    Cache for trained model artifacts.

    Provides hash-based caching for VAE, Scorer, and GP models.
    When the same configuration is used, trained models are loaded
    from cache instead of retraining.

    This is especially useful when:
    - Tuning Inference phase (only ucb_beta changes, VAE+GP stay same)
    - Running multiple trials with similar configs
    - Resuming experiments after interruption
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories for each artifact type
        self.vae_cache = self.cache_dir / "vae"
        self.scorer_cache = self.cache_dir / "scorer"
        self.gp_cache = self.cache_dir / "gp"

        for d in [self.vae_cache, self.scorer_cache, self.gp_cache]:
            d.mkdir(exist_ok=True)

        # Index for fast lookups
        self._index_path = self.cache_dir / "cache_index.json"
        self._index: Dict[str, Dict[str, CacheEntry]] = self._load_index()

        # Statistics
        self._hits = 0
        self._misses = 0

    def _load_index(self) -> Dict[str, Dict[str, CacheEntry]]:
        """Load cache index from disk."""
        if not self._index_path.exists():
            return {"vae": {}, "scorer": {}, "gp": {}}

        try:
            with open(self._index_path) as f:
                data = json.load(f)
            return {
                artifact_type: {
                    hash_key: CacheEntry.from_dict(entry)
                    for hash_key, entry in entries.items()
                }
                for artifact_type, entries in data.items()
            }
        except json.JSONDecodeError as e:
            logger.error(
                f"Cache index is corrupted at {self._index_path}: {e}. "
                "Consider deleting the cache directory to rebuild."
            )
            return {"vae": {}, "scorer": {}, "gp": {}}
        except PermissionError as e:
            logger.error(f"Permission denied reading cache index at {self._index_path}: {e}")
            return {"vae": {}, "scorer": {}, "gp": {}}
        except Exception as e:
            logger.error(f"Unexpected error loading cache index: {e}")
            return {"vae": {}, "scorer": {}, "gp": {}}

    def _save_index(self):
        """Save cache index to disk."""
        data = {
            artifact_type: {
                hash_key: entry.to_dict()
                for hash_key, entry in entries.items()
            }
            for artifact_type, entries in self._index.items()
        }
        with open(self._index_path, "w") as f:
            json.dump(data, f, indent=2)

    # ==================== VAE Caching ====================

    def get_vae_hash(self, config: Dict[str, Any]) -> str:
        """Compute hash for VAE configuration."""
        return compute_config_hash(config, VAE_AFFECTING_PARAMS)

    def has_vae(self, config: Dict[str, Any]) -> bool:
        """Check if VAE with this config is cached."""
        hash_key = self.get_vae_hash(config)
        if hash_key in self._index["vae"]:
            # Verify file still exists
            entry = self._index["vae"][hash_key]
            if Path(entry.path).exists():
                return True
            else:
                # File deleted, remove from index
                del self._index["vae"][hash_key]
                self._save_index()
        return False

    def save_vae(
        self,
        config: Dict[str, Any],
        state_dict: Dict,
        metrics: Dict[str, float],
    ):
        """
        Save VAE checkpoint to cache.

        Args:
            config: Full hyperparameter config
            state_dict: VAE state dict (vae.state_dict())
            metrics: Evaluation metrics (e.g., {"reconstruction_cosine": 0.92})
        """
        hash_key = self.get_vae_hash(config)
        path = self.vae_cache / f"{hash_key}.pt"

        # Save checkpoint
        torch.save({
            "state_dict": state_dict,
            "metrics": metrics,
            "config_hash": hash_key,
        }, path)

        # Extract config subset for debugging
        config_subset = {k: config.get(k) for k in VAE_AFFECTING_PARAMS if k in config}

        # Update index
        entry = CacheEntry(
            hash_key=hash_key,
            artifact_type="vae",
            path=str(path),
            metrics=metrics,
            config_subset=config_subset,
            created_at=datetime.now().isoformat(),
        )
        self._index["vae"][hash_key] = entry
        self._save_index()

        logger.info(f"Cached VAE artifact: {hash_key} (metrics: {metrics})")

    def load_vae(
        self,
        config: Dict[str, Any],
    ) -> Tuple[Optional[Dict], Optional[Dict[str, float]]]:
        """
        Load VAE from cache if available.

        Returns:
            (state_dict, metrics) if cached, (None, None) otherwise
        """
        hash_key = self.get_vae_hash(config)

        if hash_key not in self._index["vae"]:
            self._misses += 1
            return None, None

        entry = self._index["vae"][hash_key]
        path = Path(entry.path)

        if not path.exists():
            # File was deleted
            del self._index["vae"][hash_key]
            self._save_index()
            self._misses += 1
            return None, None

        try:
            checkpoint = torch.load(path, map_location="cpu")

            # Update access statistics
            entry.accessed_count += 1
            entry.last_accessed = datetime.now().isoformat()
            self._save_index()

            self._hits += 1
            logger.info(f"Loaded cached VAE: {hash_key} (hit #{entry.accessed_count})")

            return checkpoint["state_dict"], checkpoint["metrics"]

        except (RuntimeError, pickle.UnpicklingError) as e:
            logger.error(
                f"Failed to load cached VAE checkpoint at {path}: {e}. "
                "The checkpoint may be corrupted or incompatible. "
                "Consider clearing the cache with cache.clear('vae')."
            )
            self._misses += 1
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error loading VAE from {path}: {e}")
            self._misses += 1
            return None, None

    # ==================== GP Caching ====================

    def get_gp_hash(self, config: Dict[str, Any]) -> str:
        """Compute hash for GP configuration (includes VAE params)."""
        return compute_config_hash(config, GP_AFFECTING_PARAMS)

    def has_gp(self, config: Dict[str, Any]) -> bool:
        """Check if GP with this config is cached."""
        hash_key = self.get_gp_hash(config)
        if hash_key in self._index["gp"]:
            entry = self._index["gp"][hash_key]
            if Path(entry.path).exists():
                return True
            else:
                del self._index["gp"][hash_key]
                self._save_index()
        return False

    def save_gp(
        self,
        config: Dict[str, Any],
        gp_model: Any,
        metrics: Dict[str, float],
    ):
        """
        Save GP model to cache (pickled).

        Args:
            config: Full hyperparameter config
            gp_model: Trained GP model
            metrics: Evaluation metrics (e.g., {"spearman": 0.45})
        """
        hash_key = self.get_gp_hash(config)
        path = self.gp_cache / f"{hash_key}.pkl"

        with open(path, "wb") as f:
            pickle.dump({
                "model": gp_model,
                "metrics": metrics,
                "config_hash": hash_key,
            }, f)

        config_subset = {k: config.get(k) for k in GP_AFFECTING_PARAMS if k in config}

        entry = CacheEntry(
            hash_key=hash_key,
            artifact_type="gp",
            path=str(path),
            metrics=metrics,
            config_subset=config_subset,
            created_at=datetime.now().isoformat(),
        )
        self._index["gp"][hash_key] = entry
        self._save_index()

        logger.info(f"Cached GP artifact: {hash_key} (metrics: {metrics})")

    def load_gp(
        self,
        config: Dict[str, Any],
    ) -> Tuple[Optional[Any], Optional[Dict[str, float]]]:
        """
        Load GP from cache if available.

        Returns:
            (gp_model, metrics) if cached, (None, None) otherwise
        """
        hash_key = self.get_gp_hash(config)

        if hash_key not in self._index["gp"]:
            self._misses += 1
            return None, None

        entry = self._index["gp"][hash_key]
        path = Path(entry.path)

        if not path.exists():
            del self._index["gp"][hash_key]
            self._save_index()
            self._misses += 1
            return None, None

        try:
            with open(path, "rb") as f:
                checkpoint = pickle.load(f)

            entry.accessed_count += 1
            entry.last_accessed = datetime.now().isoformat()
            self._save_index()

            self._hits += 1
            logger.info(f"Loaded cached GP: {hash_key} (hit #{entry.accessed_count})")

            return checkpoint["model"], checkpoint["metrics"]

        except pickle.UnpicklingError as e:
            logger.error(
                f"Failed to unpickle cached GP at {path}: {e}. "
                "The checkpoint may be corrupted. "
                "Consider clearing the cache with cache.clear('gp')."
            )
            self._misses += 1
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error loading GP from {path}: {e}")
            self._misses += 1
            return None, None

    # ==================== Utilities ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "vae_cached": len(self._index["vae"]),
            "gp_cached": len(self._index["gp"]),
            "scorer_cached": len(self._index["scorer"]),
            "total_hits": self._hits,
            "total_misses": self._misses,
            "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0,
            "cache_dir": str(self.cache_dir),
        }

    def clear(self, artifact_type: Optional[str] = None):
        """
        Clear cache.

        Args:
            artifact_type: "vae", "gp", or None for all
        """
        if artifact_type is None:
            types = ["vae", "scorer", "gp"]
        else:
            types = [artifact_type]

        for atype in types:
            for entry in self._index.get(atype, {}).values():
                path = Path(entry.path)
                if path.exists():
                    path.unlink()
            self._index[atype] = {}

        self._save_index()
        logger.info(f"Cleared cache for: {types}")

    def get_best_available_vae(
        self,
        metric_name: str = "vae_reconstruction_cosine",
    ) -> Tuple[Optional[Dict], Optional[Dict[str, float]], Optional[str]]:
        """
        Get best available VAE from cache (for fallback).

        Returns:
            (state_dict, metrics, hash_key) for best VAE, or (None, None, None)
        """
        best_entry = None
        best_value = float('-inf')

        for hash_key, entry in self._index["vae"].items():
            value = entry.metrics.get(metric_name, float('-inf'))
            if value > best_value and Path(entry.path).exists():
                best_value = value
                best_entry = entry

        if best_entry is None:
            return None, None, None

        try:
            checkpoint = torch.load(best_entry.path, map_location="cpu")
            return checkpoint["state_dict"], checkpoint["metrics"], best_entry.hash_key
        except Exception as e:
            logger.error(
                f"Failed to load best available VAE from {best_entry.path}: {e}. "
                "The checkpoint may be corrupted."
            )
            return None, None, None
