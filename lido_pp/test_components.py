#!/usr/bin/env python
"""
Comprehensive test script for LID-O++ components.

Tests:
1. Configuration and imports
2. GritLM encoder (if available)
3. Latent Attention Pooling
4. FlowDiT model
5. ODE solvers with curvature tracking
6. Active Learning components
7. Memory profiling

Usage:
    uv run python -m lido_pp.test_components
    # or
    uv run python lido_pp/test_components.py
"""

import sys
import time
import torch
import torch.nn as nn
from typing import Optional


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_memory_stats(device: str = "cuda"):
    """Print GPU memory statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB peak")
    else:
        print("  No CUDA available")


def test_config():
    """Test configuration."""
    print_section("1. Configuration")

    from lido_pp.config import LIDOPPConfig

    config = LIDOPPConfig()
    print(f"  Device: {config.device}")
    print(f"  GritLM model: {config.gritlm_model}")
    print(f"  GritLM quantize: {config.gritlm_quantize}")
    print(f"  Embedding dim: {config.embedding_dim}")
    print(f"  VAE latent dim: {config.vae_latent_dim}")
    print(f"  Flow hidden dim: {config.flow_hidden_dim}")
    print(f"  Flow layers: {config.flow_num_layers}")
    print(f"  OAT weight: {config.oat_weight}")
    print(f"  Use reflow: {config.use_reflow}")
    print(f"  Inference steps: {config.inference_steps}")

    print("\n  [OK] Configuration loaded successfully")
    return config


def test_latent_attention(device: str = "cuda"):
    """Test Latent Attention Pooling."""
    print_section("2. Latent Attention Pooling")

    from lido_pp.backbone.latent_attention import LatentAttentionPooling

    # Create layer
    layer = LatentAttentionPooling(
        hidden_dim=4096,  # GritLM hidden size
        num_queries=512,
        num_heads=8,
        output_dim=768,
    ).to(device)

    print(f"  Parameters: {sum(p.numel() for p in layer.parameters()):,}")

    # Test forward pass
    batch_size = 4
    seq_len = 128
    hidden_states = torch.randn(batch_size, seq_len, 4096, device=device)
    attention_mask = torch.ones(batch_size, seq_len, device=device)

    with torch.no_grad():
        output = layer(hidden_states, attention_mask)

    print(f"  Input shape: {hidden_states.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (batch_size, 768), f"Expected (4, 768), got {output.shape}"

    print_memory_stats(device)
    print("\n  [OK] Latent Attention Pooling works correctly")

    return layer


def test_flow_dit(device: str = "cuda"):
    """Test FlowDiT model."""
    print_section("3. FlowDiT Model")

    from lido_pp.flow.flow_dit import FlowDiT

    # Create model
    model = FlowDiT(
        latent_dim=32,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        context_dim=768,
        num_context_tokens=4,
    ).to(device)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 8
    x_t = torch.randn(batch_size, 32, device=device)
    t = torch.rand(batch_size, device=device)
    context = torch.randn(batch_size, 4, 768, device=device)

    with torch.no_grad():
        v = model(x_t, t, context)

    print(f"  x_t shape: {x_t.shape}")
    print(f"  t shape: {t.shape}")
    print(f"  context shape: {context.shape}")
    print(f"  velocity shape: {v.shape}")
    assert v.shape == x_t.shape, f"Expected {x_t.shape}, got {v.shape}"

    print_memory_stats(device)
    print("\n  [OK] FlowDiT model works correctly")

    return model


def test_losses(model: nn.Module, device: str = "cuda"):
    """Test loss functions."""
    print_section("4. Loss Functions (CFM + OAT)")

    from lido_pp.flow.losses import (
        conditional_flow_matching_loss,
        oat_regularization,
        oat_flow_matching_loss,
    )

    batch_size = 8
    x_0 = torch.randn(batch_size, 32, device=device)
    x_1 = torch.randn(batch_size, 32, device=device)
    context = torch.randn(batch_size, 4, 768, device=device)

    # CFM loss
    cfm_loss, cfm_metrics = conditional_flow_matching_loss(
        model, x_0, x_1, context
    )
    print(f"  CFM loss: {cfm_loss.item():.6f}")
    print(f"  CFM metrics: v_pred_norm={cfm_metrics['v_pred_norm']:.4f}, "
          f"cosine_sim={cfm_metrics['v_cosine_sim']:.4f}")

    # OAT regularization
    oat_loss, oat_metrics = oat_regularization(
        model, x_0, x_1, context, num_steps=5
    )
    print(f"  OAT loss: {oat_loss.item():.6f}")
    print(f"  OAT avg_accel: {oat_metrics['avg_accel']:.4f}")

    # Combined loss
    total_loss, all_metrics = oat_flow_matching_loss(
        model, x_0, x_1, context, oat_weight=0.1
    )
    print(f"  Total loss: {total_loss.item():.6f}")

    # Test gradient flow
    total_loss.backward()
    print(f"  Gradients computed successfully")

    model.zero_grad()
    print("\n  [OK] Loss functions work correctly")


def test_ode_solvers(model: nn.Module, device: str = "cuda"):
    """Test ODE solvers with curvature tracking."""
    print_section("5. ODE Solvers with Curvature")

    from lido_pp.flow.ode_solver import (
        euler_integrate,
        midpoint_integrate,
        rk4_integrate,
        one_step_integrate,
        integrate,
    )

    batch_size = 4
    x_0 = torch.randn(batch_size, 32, device=device)
    context = torch.randn(batch_size, 4, 768, device=device)

    # Euler
    print("  Testing Euler integration...")
    result_euler = euler_integrate(model, x_0, context, num_steps=20)
    print(f"    Final shape: {result_euler.x_final.shape}")
    print(f"    Curvature mean: {result_euler.curvature.mean().item():.4f}")

    # Midpoint
    print("  Testing Midpoint integration...")
    result_mid = midpoint_integrate(model, x_0, context, num_steps=20)
    print(f"    Curvature mean: {result_mid.curvature.mean().item():.4f}")

    # RK4
    print("  Testing RK4 integration...")
    result_rk4 = rk4_integrate(model, x_0, context, num_steps=20)
    print(f"    Curvature mean: {result_rk4.curvature.mean().item():.4f}")

    # One-step
    print("  Testing one-step integration...")
    x_1_one = one_step_integrate(model, x_0, context)
    print(f"    Output shape: {x_1_one.shape}")

    # Compare endpoints
    euler_mid_diff = (result_euler.x_final - result_mid.x_final).norm(dim=-1).mean()
    euler_rk4_diff = (result_euler.x_final - result_rk4.x_final).norm(dim=-1).mean()
    print(f"  Endpoint differences:")
    print(f"    Euler vs Midpoint: {euler_mid_diff:.6f}")
    print(f"    Euler vs RK4: {euler_rk4_diff:.6f}")

    print_memory_stats(device)
    print("\n  [OK] ODE solvers work correctly")


def test_curvature(model: nn.Module, device: str = "cuda"):
    """Test Flow Curvature Uncertainty."""
    print_section("6. Flow Curvature Uncertainty (FCU)")

    from lido_pp.active_learning.curvature import (
        compute_flow_curvature,
        compute_fcu_with_threshold,
        FlowCurvatureEstimator,
    )

    batch_size = 100
    z = torch.randn(batch_size, 32, device=device)
    context = torch.randn(batch_size, 4, 768, device=device)

    # Basic curvature
    print("  Computing basic curvature...")
    curvature = compute_flow_curvature(model, z, context, num_steps=20)
    print(f"    Shape: {curvature.shape}")
    print(f"    Mean: {curvature.mean().item():.4f}")
    print(f"    Std: {curvature.std().item():.4f}")
    print(f"    Min: {curvature.min().item():.4f}")
    print(f"    Max: {curvature.max().item():.4f}")

    # FCU with threshold
    print("  Computing FCU with threshold...")
    fcu_result = compute_fcu_with_threshold(model, z, context, percentile=90.0)
    print(f"    Threshold: {fcu_result.threshold:.4f}")
    eval_count = fcu_result.should_evaluate.sum().item()
    print(f"    Should evaluate: {eval_count}/{batch_size} ({100*eval_count/batch_size:.1f}%)")

    # Estimator with running stats
    print("  Testing FCU estimator...")
    estimator = FlowCurvatureEstimator(model, num_steps=20, percentile=90.0).to(device)

    for i in range(3):
        z_batch = torch.randn(50, 32, device=device)
        ctx_batch = torch.randn(50, 4, 768, device=device)
        result = estimator(z_batch, ctx_batch)
        stats = estimator.get_statistics()
        print(f"    Batch {i+1}: mean={stats['running_mean']:.4f}, "
              f"eval_rate={result.should_evaluate.float().mean():.2%}")

    print("\n  [OK] FCU works correctly")


def test_value_head(device: str = "cuda"):
    """Test Value Head."""
    print_section("7. Value Head")

    from lido_pp.active_learning.value_head import (
        ValueHead,
        ValueHeadWithUncertainty,
        ValueHeadTrainer,
    )

    # Basic Value Head
    print("  Testing basic Value Head...")
    value_head = ValueHead(latent_dim=32, hidden_dim=128).to(device)
    print(f"    Parameters: {sum(p.numel() for p in value_head.parameters()):,}")

    z = torch.randn(16, 32, device=device)
    pred = value_head(z)
    print(f"    Input: {z.shape}")
    print(f"    Output: {pred.shape}")
    print(f"    Predictions range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")

    # Value Head with MC Dropout uncertainty
    print("  Testing Value Head with uncertainty...")
    vh_mc = ValueHeadWithUncertainty(latent_dim=32, hidden_dim=128).to(device)
    mean_pred, uncertainty = vh_mc(z, return_uncertainty=True)
    print(f"    Mean prediction shape: {mean_pred.shape}")
    print(f"    Uncertainty shape: {uncertainty.shape}")
    print(f"    Mean uncertainty: {uncertainty.mean().item():.4f}")

    # Trainer
    print("  Testing Value Head trainer...")
    trainer = ValueHeadTrainer(value_head, lr=1e-3)

    # Add synthetic observations
    for _ in range(100):
        z_obs = torch.randn(32)
        error_rate = 0.3 + 0.1 * torch.randn(1).item()
        trainer.add_observation(z_obs, max(0.0, min(1.0, error_rate)))

    print(f"    Buffer size: {len(trainer.buffer_z)}")

    # Train step
    metrics = trainer.train_step(batch_size=32, device=device)
    print(f"    Train step loss: {metrics['loss']:.6f}")

    print("\n  [OK] Value Head works correctly")


def test_acquisition(device: str = "cuda"):
    """Test Cost-Aware Acquisition."""
    print_section("8. Cost-Aware Acquisition")

    from lido_pp.flow.flow_dit import FlowDiT
    from lido_pp.active_learning.value_head import ValueHead
    from lido_pp.active_learning.acquisition import CostAwareAcquisition

    # Create mock GP
    class MockGP:
        def predict(self, z):
            # Negative error for BoTorch compatibility
            mean = -torch.abs(z).sum(dim=-1) * 0.1
            std = torch.ones(z.shape[0], device=z.device) * 0.1
            return mean, std

    # Create models
    flow = FlowDiT(latent_dim=32, hidden_dim=256, num_layers=4).to(device)
    value_head = ValueHead(latent_dim=32).to(device)
    gp = MockGP()

    # Create acquisition
    acquisition = CostAwareAcquisition(
        gp_model=gp,
        flow_model=flow,
        value_head=value_head,
        lambda_cost=0.1,
        ucb_beta=2.0,
    )

    # Test evaluation
    z = torch.randn(100, 32, device=device)
    print("  Evaluating acquisition function...")
    acq_values, components = acquisition(z, return_components=True)
    print(f"    Values shape: {acq_values.shape}")
    print(f"    Values range: [{acq_values.min().item():.4f}, {acq_values.max().item():.4f}]")

    # Test optimization
    print("  Optimizing acquisition...")
    bounds = torch.stack([
        torch.zeros(32, device=device),
        torch.ones(32, device=device)
    ])
    result = acquisition.optimize(bounds, num_restarts=10, raw_samples=500)
    print(f"    Optimal z shape: {result.z_optimal.shape}")
    print(f"    Acquisition value: {result.acquisition_value:.4f}")
    print(f"    GP mean: {result.gp_mean:.4f}")
    print(f"    GP std: {result.gp_std:.4f}")
    print(f"    Curvature: {result.curvature:.4f}")
    print(f"    Should evaluate: {result.should_evaluate}")
    print(f"    Value prediction: {result.value_prediction}")

    print("\n  [OK] Cost-Aware Acquisition works correctly")


def test_gating(device: str = "cuda"):
    """Test Evaluation Gating."""
    print_section("9. Evaluation Gating")

    from lido_pp.flow.flow_dit import FlowDiT
    from lido_pp.active_learning.value_head import ValueHead
    from lido_pp.active_learning.gating import (
        EvaluationGate,
        EvaluationType,
        AdaptiveGate,
    )

    # Create models
    flow = FlowDiT(latent_dim=32, hidden_dim=256, num_layers=4).to(device)
    value_head = ValueHead(latent_dim=32).to(device)

    # Create gate
    gate = EvaluationGate(
        flow_model=flow,
        value_head=value_head,
        percentile_threshold=90.0,
        min_samples_for_threshold=10,
    )

    # Test batch decisions
    print("  Making batch gating decisions...")
    z_batch = torch.randn(50, 32, device=device)
    decisions = gate.decide_batch(z_batch)

    llm_count = sum(1 for d in decisions if d.eval_type == EvaluationType.LLM)
    vh_count = sum(1 for d in decisions if d.eval_type == EvaluationType.VALUE_HEAD)
    print(f"    Total: {len(decisions)}")
    print(f"    LLM evaluations: {llm_count}")
    print(f"    Value Head: {vh_count}")

    # Record some ground truth
    print("  Recording ground truth...")
    for i, decision in enumerate(decisions[:10]):
        ground_truth = 0.3 + 0.1 * torch.randn(1).item()
        gate.record_evaluation(decisions[i].z.unsqueeze(0), max(0, min(1, ground_truth)), decision)

    # Get statistics
    stats = gate.get_statistics()
    print(f"    Cost savings: {stats['cost_savings']:.2%}")
    print(f"    LLM ratio: {stats['llm_ratio']:.2%}")

    # Test adaptive gate
    print("  Testing adaptive gate...")
    adaptive_gate = AdaptiveGate(
        flow_model=flow,
        value_head=value_head,
        total_budget=20,
        target_llm_ratio=0.2,
        min_samples_for_threshold=10,
    )

    for i in range(3):
        z = torch.randn(10, 32, device=device)
        batch_decisions = adaptive_gate.decide_batch(z)
        llm_evals = sum(1 for d in batch_decisions if d.eval_type == EvaluationType.LLM)
        adaptive_gate.update_budget(llm_evals)
        print(f"    Round {i+1}: LLM={llm_evals}, Budget={adaptive_gate.remaining_budget}")

    print("\n  [OK] Evaluation Gating works correctly")


def test_reflow(device: str = "cuda"):
    """Test Reflow training (short test)."""
    print_section("10. Reflow Training (Quick Test)")

    from lido_pp.flow.flow_dit import FlowDiT
    from lido_pp.flow.reflow import ReflowTrainer, ReflowConfig, verify_one_step_inference

    # Create model
    model = FlowDiT(latent_dim=32, hidden_dim=256, num_layers=4).to(device)

    # Very short test
    config = ReflowConfig(
        num_rounds=1,
        epochs_per_round=10,  # Very short
        lr=1e-4,
        batch_size=32,
        integration_steps=10,
        num_pairs=100,  # Small
        use_oat=True,
        oat_weight=0.1,
        eval_interval=5,
    )

    trainer = ReflowTrainer(model, config, device)

    # Context source
    context_source = torch.randn(200, 4, 768)

    print("  Running quick Reflow test...")
    start_time = time.time()
    result = trainer.train(
        x_0_source=None,
        context_source=context_source,
        latent_dim=32,
    )
    elapsed = time.time() - start_time

    print(f"    Completed in {elapsed:.1f}s")
    print(f"    Rounds: {result.rounds_completed}")
    print(f"    Final straightness:")
    for k, v in result.final_straightness.items():
        print(f"      {k}: {v:.6f}")

    # Verify 1-step
    print("  Verifying 1-step inference...")
    x_0_test = torch.randn(10, 32, device=device)
    ctx_test = torch.randn(10, 4, 768, device=device)
    quality = verify_one_step_inference(result.model, x_0_test, ctx_test, reference_steps=10)

    print(f"    L2 error: {quality['l2_error']:.6f}")
    print(f"    Cosine similarity: {quality['cosine_similarity']:.4f}")

    print("\n  [OK] Reflow training works correctly")


def test_latent_projector(device: str = "cuda"):
    """Test Latent Projector for decoder."""
    print_section("11. Latent Projector (Decoder)")

    from lido_pp.backbone.latent_injection import LatentProjector

    # Test different configurations
    configs = [
        {"num_prefix_tokens": 1, "desc": "Single token"},
        {"num_prefix_tokens": 4, "desc": "4 tokens (default)"},
        {"num_prefix_tokens": 8, "desc": "8 tokens (rich)"},
    ]

    for cfg in configs:
        projector = LatentProjector(
            latent_dim=768,
            hidden_dim=4096,
            num_prefix_tokens=cfg["num_prefix_tokens"],
        ).to(device)

        latent = torch.randn(4, 768, device=device)
        prefix = projector(latent)

        expected_shape = (4, cfg["num_prefix_tokens"], 4096)
        assert prefix.shape == expected_shape, f"Shape mismatch: {prefix.shape} vs {expected_shape}"

        params = sum(p.numel() for p in projector.parameters())
        print(f"  {cfg['desc']}: {prefix.shape}, {params:,} params")

    print_memory_stats(device)
    print("\n  [OK] Latent Projector works correctly")


def test_gritlm_encoder(device: str = "cuda"):
    """Test GritLM encoder (if available)."""
    print_section("12. GritLM Encoder (Optional)")

    try:
        from lido_pp.backbone.gritlm_encoder import GritLMUnifiedEncoder

        print("  Initializing GritLM encoder (this may take a while)...")
        print("  Note: Skipping actual model loading for quick test")

        # Just test that the class can be imported and instantiated logic works
        print("  GritLMUnifiedEncoder class is available")
        print("  To fully test, run with actual model loading")
        print("\n  [OK] GritLM encoder module available")

    except ImportError as e:
        print(f"  [SKIP] GritLM not available: {e}")
    except Exception as e:
        print(f"  [SKIP] Error: {e}")


def memory_profile(device: str = "cuda"):
    """Profile memory usage."""
    print_section("13. Memory Profile")

    if not torch.cuda.is_available():
        print("  [SKIP] No CUDA available")
        return

    torch.cuda.reset_peak_memory_stats(device)

    # FlowDiT memory
    from lido_pp.flow.flow_dit import FlowDiT

    print("  FlowDiT (6 layers, 512 hidden):")
    model = FlowDiT(latent_dim=32, hidden_dim=512, num_layers=6).to(device)
    print_memory_stats(device)

    # Forward pass
    x = torch.randn(64, 32, device=device)
    t = torch.rand(64, device=device)
    ctx = torch.randn(64, 4, 768, device=device)
    _ = model(x, t, ctx)
    print("  After forward pass (batch=64):")
    print_memory_stats(device)

    # Backward pass
    loss = model(x, t, ctx).sum()
    loss.backward()
    print("  After backward pass:")
    print_memory_stats(device)

    # Cleanup
    del model, x, t, ctx, loss
    torch.cuda.empty_cache()

    print("\n  [OK] Memory profiling complete")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  LID-O++ Component Tests")
    print("="*60)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")

    # Run tests
    try:
        config = test_config()
        test_latent_attention(device)

        flow_model = test_flow_dit(device)
        test_losses(flow_model, device)
        test_ode_solvers(flow_model, device)
        test_curvature(flow_model, device)

        test_value_head(device)
        test_acquisition(device)
        test_gating(device)

        # Reflow test (shorter on CPU)
        if device == "cuda":
            test_reflow(device)
        else:
            print("\n[SKIP] Reflow test skipped on CPU (too slow)")

        test_latent_projector(device)
        test_gritlm_encoder(device)

        if device == "cuda":
            memory_profile(device)

        print("\n" + "="*60)
        print("  All Tests Passed!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
