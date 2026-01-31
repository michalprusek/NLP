# EcoFlow Pipeline

Flow matching + Bayesian optimization for prompt optimization in SONAR embedding space.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EcoFlow BO Loop                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. GP-UCB Optimization          2. Flow Projection        3. Evaluate    │
│   ┌─────────────────────┐        ┌─────────────────┐       ┌────────────┐  │
│   │ z* = argmax UCB(z)  │   ──►  │ encode → decode │  ──►  │ SONAR dec  │  │
│   │ gradient ascent     │        │ manifold proj.  │       │ LLM eval   │  │
│   └─────────────────────┘        └─────────────────┘       └────────────┘  │
│            │                             │                        │        │
│            └─────────────────────────────┴────────────────────────┘        │
│                                    │                                       │
│                           4. Update GP                                     │
│                           ┌─────────────────┐                              │
│                           │ gp.update(z, y) │                              │
│                           └─────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Algorithm

Each iteration:
1. **GP-UCB Optimization**: Find `z* = argmax [μ(z) + α·σ(z)]` via gradient ascent
2. **Flow Projection**: Encode z* to noise space, decode back to ensure on-manifold
3. **Decode & Evaluate**: SONAR decoder → text prompt → LLM evaluation on GSM8K
4. **Update GP**: Add (z*, score) observation, refit GP

## File Structure

```
ecoflow/
├── __init__.py              # Package exports
├── run.py                   # CLI entry point
│
├── # Core BO Pipeline
├── optimization_loop.py     # BOOptimizationLoop - main orchestrator
├── gp_surrogate.py          # GP surrogates (MSR, BAxUS, Heteroscedastic)
├── guided_flow.py           # GuidedFlowSampler with sample_optimal()
│
├── # Flow Model
├── velocity_network.py      # DiT-style velocity network with AdaLN
├── flow_model.py            # FlowMatchingModel (ODE sampling)
├── train_flow.py            # Training script with OT-CFM
│
├── # Utilities
├── decoder.py               # SonarDecoder (embedding → text)
├── data.py                  # Dataset and dataloader
├── batch_selection.py       # Batch candidate selection (optional)
└── validate.py              # Model validation utilities
```

## Key Classes

### `SonarGPSurrogate` / `BAxUSGPSurrogate`
GP surrogate with MSR initialization for 1024D optimization.

```python
gp = SonarGPSurrogate(D=1024, device='cuda')
gp.fit(X, Y)                           # Fit GP
mean, std = gp.predict(X_new)          # Predict
z_opt, ucb = gp.optimize_ucb(alpha=1.96)  # Find optimal point
```

### `GuidedFlowSampler`
Generates samples with optional UCB guidance.

```python
sampler = GuidedFlowSampler(flow_model, gp)

# Single optimal candidate (main method)
z, info = sampler.sample_optimal(device='cuda')

# Guided sampling from noise (exploration)
z = sampler.sample(n_samples=10)
```

### `BOOptimizationLoop`
Main optimization orchestrator.

```python
loop = BOOptimizationLoop(flow_model, gp, sampler, decoder, evaluator, llm)
loop.warm_start('embeddings.pt')  # Initialize from pre-evaluated data

for _ in range(100):
    result = loop.step()
    print(f"Iter {result['iteration']}: {result['score']:.3f}")
```

## Usage

### Training Flow Model

```bash
uv run python -m ecoflow.train_flow \
  --data datasets/sonar_embeddings.pt \
  --epochs 100 --batch-size 1024
```

### Running Optimization

```bash
uv run python -m ecoflow.run \
  --flow-checkpoint path/to/flow.pt \
  --warm-start datasets/evaluated_instructions.pt \
  --iterations 100
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--ucb-alpha` | UCB exploration weight | 1.96 |
| `--n-restarts` | GP optimization restarts | 5 |
| `--n-opt-steps` | Gradient steps per restart | 100 |
| `--num-steps` | ODE integration steps | 50 |
| `--guidance-strength` | λ for guided sampling | 1.0 |

## Design Principles

1. **Single candidate per iteration**: GP-UCB optimization produces one optimal point
2. **Flow as manifold projector**: encode/decode ensures samples stay on SONAR distribution
3. **Simple over complex**: No hybrid strategies, no batch selection complexity
4. **Minimal dependencies**: Core pipeline uses only GP, flow model, decoder, evaluator
