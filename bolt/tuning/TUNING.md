# BOLT Hyperparameter Tuning System

## Overview

Sophisticated hyperparameter optimization wrapper for the BOLT pipeline using **Coordinate Descent** strategy with dual-GPU parallel execution.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CoordinateDescentTuner                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Phase 1: VAE     → Phase 2: Scorer → Phase 3: GP → Phase 4 │ │
│  │ (Reconstruction)   (Selection)       (Prediction) (Inference)│
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  DualGPUExecutor                            │ │
│  │  ┌─────────────┐              ┌─────────────┐              │ │
│  │  │   GPU 0     │              │   GPU 1     │              │ │
│  │  │  (L40S)     │    Queue     │  (L40S)     │              │ │
│  │  │  48GB VRAM  │ ←─────────→  │  48GB VRAM  │              │ │
│  │  └─────────────┘              └─────────────┘              │ │
│  │         ↑                            ↑                      │ │
│  │    ProcessPoolExecutor (spawn context for CUDA)             │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Coordinate Descent Strategy

Each phase optimizes one component while keeping others fixed:

### Phase 1: VAE Optimization
- **Objective**: Maximize Retrieval Accuracy @8
- **Fixed**: GP params, inference params
- **Tune**: `vae_beta`, `vae_mse_weight`, latent dims, `vae_lr`
- **Checkpoint Gate**: retrieval_acc ≥ 0.85, lipschitz < 10

### Phase 2: Scorer Optimization (BOLT only)
- **Objective**: Maximize NDCG@8
- **Fixed**: VAE (from Phase 1), GP params
- **Tune**: `selection_weight`, `mmr_lambda`, Set Transformer params
- **Checkpoint Gate**: NDCG ≥ 0.70

### Phase 3: GP Optimization
- **Objective**: Maximize Spearman Correlation
- **Fixed**: VAE, Scorer
- **Tune**: `gp_lr`, `use_deep_kernel`, lengthscale priors
- **Checkpoint Gate**: spearman ≥ 0.40

### Phase 4: Inference Optimization
- **Objective**: Maximize Final Accuracy
- **Fixed**: VAE, Scorer, GP
- **Tune**: `ucb_beta`, `num_restarts`, `distance_weight`
- **Checkpoint Gate**: accuracy ≥ 91.5%

## Parameter Tiers

### Tier 1: CRITICAL (tune first)
```python
{
    "vae_beta": (0.005, 0.1),        # KL weight
    "vae_mse_weight": (0.1, 0.5),    # MSE weight
    "instruction_latent_dim": [12, 16, 24, 32],
    "exemplar_latent_dim": [8, 12, 16, 24],
    "ucb_beta": (4.0, 16.0),         # Exploration
    "mmr_lambda": (0.3, 0.9),        # Diversity
    "selection_weight": (0.05, 0.4), # Scorer weight
}
```

### Tier 2: IMPORTANT (tune after Tier 1 stable)
```python
{
    "set_transformer_heads": [4, 8],
    "set_transformer_hidden": [64, 128, 256],
    "num_inducing_points": [4, 8],
    "gp_lr": (0.001, 0.01),
    "use_deep_kernel": [True, False],
    "vae_lr": (0.0001, 0.001),
    "vae_epochs": [20000, 50000, 100000],
}
```

### Tier 3: FINETUNE
```python
{
    "num_restarts": [32, 64, 128],
    "raw_samples": [2048, 4096, 8192],
    "cosine_sim_threshold": (0.85, 0.95),
    "distance_weight": (1.0, 4.0),
    "latent_noise_scale": (0.01, 0.1),
}
```

## Metrics Taxonomy (25 metrics)

### VAE Quality (6 metrics)
| Metric | Target | Description |
|--------|--------|-------------|
| reconstruction_cosine | ≥0.93 | Cosine similarity after encode/decode |
| reconstruction_mse | ≤0.05 | Mean squared error |
| kl_divergence | 0.1-1.0 | KL term for regularization |
| retrieval_accuracy_at_8 | ≥0.85 | Can retrieve original from latent |
| lipschitz_constant | <10 | Smoothness of latent space |
| percentile10_cosine | ≥0.70 | Worst-case reconstruction |

### Scorer Quality (5 metrics)
| Metric | Target | Description |
|--------|--------|-------------|
| ndcg_at_8 | ≥0.70 | Ranking quality |
| mrr | ≥0.50 | Mean reciprocal rank |
| selection_loss | <2.0 | ListMLE loss |
| exemplar_diversity | >0.3 | Pairwise distance |
| exemplar_variance | >0 | Variance across iterations |

### GP Quality (5 metrics)
| Metric | Target | Description |
|--------|--------|-------------|
| spearman_correlation | ≥0.40 | Rank correlation |
| rmse | <0.05 | Root mean squared error |
| calibration_error | <0.10 | Confidence vs accuracy |
| lengthscale_ratio | 0.5-2.0 | Instruction/exemplar ratio |
| nll_heldout | minimize | Negative log-likelihood |

### Optimization Quality (5 metrics)
| Metric | Target | Description |
|--------|--------|-------------|
| system_gap | <0.5 | Latent vs realized distance |
| rejection_rate | <0.30 | Invalid samples ratio |
| improvement_rate | >0.20 | Improving iterations |
| best_error_rate | <0.085 | Minimum error achieved |
| convergence_speed | <20 | Iterations to 90% best |

### End-to-End (4 metrics)
| Metric | Target | Description |
|--------|--------|-------------|
| final_accuracy | ≥91.5% | Full evaluation accuracy |
| sample_efficiency | maximize | Accuracy per LLM call |
| wall_clock_time | minimize | Total runtime |
| gpu_memory_peak | <20GB | Maximum VRAM usage |

## Dual-GPU Optimization

### Process Isolation
```python
# CUDA requires 'spawn' not 'fork' to avoid memory issues
mp_context = mp.get_context('spawn')

self._executor = ProcessPoolExecutor(
    max_workers=len(self.gpu_ids),  # One worker per GPU
    mp_context=mp_context,
)
```

### GPU Assignment
```python
def _run_trial_in_process(trial_id, config_dict, phase, output_dir, gpu_id):
    # Each process gets exclusive GPU access
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)  # Always device 0 within isolated process
```

### Load Balancing
- Priority queue for trials (higher tier = higher priority)
- Automatic retry on OOM with exponential backoff
- Heartbeat monitoring every 60s
- Checkpoint save every 10 trials or 1 hour

### Optimizations for L40S (48GB VRAM)
```python
# Recommended settings for maximum throughput:
{
    "batch_size": 128,           # Large batches fit in 48GB
    "num_inducing_points": 8,    # More inducing points possible
    "set_transformer_hidden": 256,  # Larger models
    "raw_samples": 8192,         # More acquisition samples
    "num_restarts": 128,         # More restarts
}
```

## Usage

### Quick Test (5 trials per phase)
```bash
uv run python -m bolt.tuning.run_tuning \
    --quick \
    --output-dir bolt/tuning/results/quick_test \
    --gpus 0
```

### Full Run (dual GPU)
```bash
tmux new-session -d -s bolt_tuning \
    "uv run python -m bolt.tuning.run_tuning \
        --output-dir bolt/tuning/results/full_run \
        --gpus 0,1 \
        --phases vae,scorer,gp,inference \
        2>&1 | tee bolt/tuning/results/tuning.log; exec bash"
```

### Resume from Checkpoint
```bash
uv run python -m bolt.tuning.run_tuning \
    --output-dir bolt/tuning/results/full_run \
    --gpus 0,1 \
    --resume
```

### Single Phase
```bash
uv run python -m bolt.tuning.run_tuning \
    --phases vae \
    --gpus 0,1 \
    --output-dir bolt/tuning/results/vae_only
```

## Output Structure

```
bolt/tuning/results/
├── coordinator_state.json    # Resumable state
├── coordinator.log           # Main log
├── best_config.yaml          # Final best configuration
├── tuning_report.md          # Human-readable report
├── trials/
│   ├── executor_state.json   # Executor checkpoint
│   ├── executor.log          # Executor log
│   ├── heartbeat.json        # Live status
│   ├── vae_CRITICAL_0_xxx/   # Per-trial outputs
│   │   ├── vae.pt            # Trained VAE
│   │   ├── metrics.json      # Trial metrics
│   │   └── config.yaml       # Trial config
│   └── ...
└── phase_results/
    ├── vae.json              # VAE phase summary
    ├── scorer.json
    ├── gp.json
    └── inference.json
```

## Long-Running Experiments

The system is designed for experiments lasting days/weeks:

1. **Checkpoint Recovery**: Automatic state persistence every 10 trials
2. **Heartbeat Monitoring**: Status file updated every 60s
3. **Graceful Shutdown**: Ctrl+C saves state before exit
4. **Auto-Retry**: Failed trials retry up to 3 times
5. **tmux Integration**: Run in tmux session for connection resilience

### Monitoring Long Runs
```bash
# Check live status
cat bolt/tuning/results/trials/heartbeat.json

# Follow logs
tail -f bolt/tuning/results/coordinator.log

# Check phase progress
cat bolt/tuning/results/coordinator_state.json | jq '.completed_phases'
```

## Implementation Files

| File | Purpose |
|------|---------|
| `coordinator.py` | Main orchestrator, phase sequencing |
| `parallel_executor.py` | Dual-GPU process management |
| `trial_runner.py` | Single trial execution |
| `metrics.py` | 25 metric implementations |
| `hyperspace.py` | Parameter space definitions |
| `results_tracker.py` | SQLite-backed experiment tracking |
| `run_tuning.py` | CLI entry point |
