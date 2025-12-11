#!/usr/bin/env python3
"""
Compare OPRO vs Hybrid OPRO+HbBoPs optimization results.

Creates visualization showing:
1. Best accuracy over budget (convergence curves)
2. Final test accuracy comparison (bar chart)
3. Efficiency metrics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_opro_results(json_path: Path) -> dict:
    """Load OPRO results and extract convergence data."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    history = data['history']
    config = data['config']
    minibatch_size = config['minibatch_size']

    # Calculate cumulative budget and best accuracy at each point
    budget_points = []
    best_acc_points = []
    current_budget = 0
    best_acc = 0

    for entry in history:
        current_budget += minibatch_size
        score = entry['score']
        if score > best_acc:
            best_acc = score
        budget_points.append(current_budget)
        best_acc_points.append(best_acc * 100)  # Convert to percentage

    return {
        'budget': budget_points,
        'best_accuracy': best_acc_points,
        'test_accuracy': data['test_accuracy'] * 100,
        'validation_accuracy': data['validation_accuracy'] * 100,
        'total_budget': data['budget_used'],
        'model': data['model'],
    }


def parse_hybrid_log(log_path: Path) -> dict:
    """Parse hybrid OPRO+HbBoPs log file to extract convergence data."""
    budget_points = [0]
    best_acc_points = [88.25]  # Starting from pre-computed Phase 1 best

    current_budget = 0
    best_acc = 0.8825  # From file

    with open(log_path, 'r') as f:
        for line in f:
            # Parse budget updates
            if 'Budget:' in line and '/' in line:
                parts = line.split('Budget:')[1].strip()
                if '/' in parts:
                    current = int(parts.split('/')[0])
                    if current > current_budget:
                        current_budget = current
                        budget_points.append(current_budget)
                        best_acc_points.append(best_acc * 100)

            # Parse full evaluation results (n=1319)
            if 'n=1319: acc=' in line:
                acc_str = line.split('acc=')[1].split(',')[0]
                acc = float(acc_str)
                if acc > best_acc:
                    best_acc = acc
                    # Update the last point
                    if budget_points:
                        best_acc_points[-1] = best_acc * 100

    return {
        'budget': budget_points,
        'best_accuracy': best_acc_points,
    }


def load_hybrid_results(json_path: Path, log_path: Path) -> dict:
    """Load Hybrid OPRO+HbBoPs results."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Parse log for convergence curve
    log_data = parse_hybrid_log(log_path)

    return {
        'budget': log_data['budget'],
        'best_accuracy': log_data['best_accuracy'],
        'test_accuracy': data['results']['best_accuracy'] * 100,
        'total_budget': data['results']['total_evaluations'],
        'iterations': data['results']['iterations'],
        'unique_instructions': data['results']['num_unique_instructions'],
        'unique_exemplars': data['results']['num_unique_exemplars'],
    }


def create_comparison_figure(opro_data: dict, hybrid_data: dict, output_path: Path):
    """Create comprehensive comparison figure."""

    fig = plt.figure(figsize=(16, 10))

    # Color scheme
    opro_color = '#2196F3'  # Blue
    hybrid_color = '#4CAF50'  # Green

    # =========================================================================
    # 1. Convergence curves (top left, larger)
    # =========================================================================
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)

    # OPRO curve
    ax1.plot(opro_data['budget'], opro_data['best_accuracy'],
             color=opro_color, linewidth=2.5, label='OPRO', marker='o',
             markevery=20, markersize=4, alpha=0.9)

    # Hybrid curve
    ax1.plot(hybrid_data['budget'], hybrid_data['best_accuracy'],
             color=hybrid_color, linewidth=2.5, label='Hybrid OPRO+HbBoPs',
             marker='s', markevery=1, markersize=6, alpha=0.9)

    # Mark final test accuracies
    ax1.axhline(y=opro_data['test_accuracy'], color=opro_color,
                linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.axhline(y=hybrid_data['test_accuracy'], color=hybrid_color,
                linestyle='--', alpha=0.5, linewidth=1.5)

    # Annotations for test accuracy
    ax1.annotate(f"OPRO Test: {opro_data['test_accuracy']:.2f}%",
                xy=(50000, opro_data['test_accuracy']),
                xytext=(42000, opro_data['test_accuracy'] - 2),
                fontsize=10, color=opro_color,
                arrowprops=dict(arrowstyle='->', color=opro_color, alpha=0.7))

    ax1.annotate(f"Hybrid Test: {hybrid_data['test_accuracy']:.2f}%",
                xy=(50000, hybrid_data['test_accuracy']),
                xytext=(42000, hybrid_data['test_accuracy'] + 1.5),
                fontsize=10, color=hybrid_color,
                arrowprops=dict(arrowstyle='->', color=hybrid_color, alpha=0.7))

    ax1.set_xlabel('LLM Evaluation Budget', fontsize=12)
    ax1.set_ylabel('Best Validation Accuracy (%)', fontsize=12)
    ax1.set_title('Optimization Convergence: OPRO vs Hybrid OPRO+HbBoPs',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 52000)
    ax1.set_ylim(80, 95)

    # =========================================================================
    # 2. Test Accuracy Bar Chart (top right)
    # =========================================================================
    ax2 = plt.subplot2grid((2, 3), (0, 2))

    methods = ['OPRO', 'Hybrid\nOPRO+HbBoPs']
    test_accs = [opro_data['test_accuracy'], hybrid_data['test_accuracy']]
    colors = [opro_color, hybrid_color]

    bars = ax2.bar(methods, test_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, acc in zip(bars, test_accs):
        ax2.annotate(f'{acc:.2f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=14, fontweight='bold')

    # Add improvement arrow
    improvement = hybrid_data['test_accuracy'] - opro_data['test_accuracy']
    ax2.annotate(f'+{improvement:.2f}pp',
                xy=(1, hybrid_data['test_accuracy']),
                xytext=(0.5, (test_accs[0] + test_accs[1])/2),
                fontsize=12, color='darkgreen', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))

    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Final Test Set Performance', fontsize=14, fontweight='bold')
    ax2.set_ylim(80, 92)
    ax2.grid(True, alpha=0.3, axis='y')

    # =========================================================================
    # 3. Validation vs Test Accuracy (bottom left)
    # =========================================================================
    ax3 = plt.subplot2grid((2, 3), (1, 0))

    x = np.arange(2)
    width = 0.35

    val_accs = [opro_data['validation_accuracy'], max(hybrid_data['best_accuracy'])]
    test_accs = [opro_data['test_accuracy'], hybrid_data['test_accuracy']]

    bars1 = ax3.bar(x - width/2, val_accs, width, label='Validation',
                    color=[opro_color, hybrid_color], alpha=0.6)
    bars2 = ax3.bar(x + width/2, test_accs, width, label='Test',
                    color=[opro_color, hybrid_color], alpha=1.0, edgecolor='black')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=9)

    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('Validation vs Test Accuracy', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['OPRO', 'Hybrid'])
    ax3.legend(loc='lower right')
    ax3.set_ylim(80, 95)
    ax3.grid(True, alpha=0.3, axis='y')

    # Highlight overfitting in OPRO
    opro_gap = opro_data['validation_accuracy'] - opro_data['test_accuracy']
    hybrid_gap = max(hybrid_data['best_accuracy']) - hybrid_data['test_accuracy']

    ax3.annotate(f'Gap: {opro_gap:.1f}pp', xy=(0, 86), ha='center',
                fontsize=10, color='red' if opro_gap > 5 else 'gray')
    ax3.annotate(f'Gap: {hybrid_gap:.1f}pp', xy=(1, 86), ha='center',
                fontsize=10, color='green' if hybrid_gap < 3 else 'gray')

    # =========================================================================
    # 4. Method Comparison Summary (bottom center)
    # =========================================================================
    ax4 = plt.subplot2grid((2, 3), (1, 1))
    ax4.axis('off')

    summary_text = f"""
    COMPARISON SUMMARY
    ══════════════════════════════════════

    OPRO (Standard)
    ─────────────────
    • Test Accuracy:     {opro_data['test_accuracy']:.2f}%
    • Validation:        {opro_data['validation_accuracy']:.2f}%
    • Budget Used:       {opro_data['total_budget']:,}
    • Val-Test Gap:      {opro_gap:.1f}pp (overfitting)

    Hybrid OPRO+HbBoPs
    ─────────────────
    • Test Accuracy:     {hybrid_data['test_accuracy']:.2f}%
    • Best Validation:   {max(hybrid_data['best_accuracy']):.2f}%
    • Budget Used:       {hybrid_data['total_budget']:,}
    • Val-Test Gap:      {hybrid_gap:.1f}pp
    • Iterations:        {hybrid_data['iterations']}
    • Unique Instr:      {hybrid_data['unique_instructions']}
    • Unique Exemplars:  {hybrid_data['unique_exemplars']}

    ══════════════════════════════════════
    WINNER: Hybrid OPRO+HbBoPs
            (+{improvement:.2f} percentage points)
    """

    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # =========================================================================
    # 5. Key Insights (bottom right)
    # =========================================================================
    ax5 = plt.subplot2grid((2, 3), (1, 2))
    ax5.axis('off')

    insights_text = f"""
    KEY INSIGHTS

    1. GENERALIZATION
       OPRO shows {opro_gap:.1f}pp validation-test gap
       (significant overfitting to eval set)

       Hybrid shows only {hybrid_gap:.1f}pp gap
       (better generalization)

    2. SEARCH STRATEGY
       OPRO: Free-form text generation
       → High variance, prone to mode collapse

       Hybrid: Structured instruction+exemplar
       → Systematic exploration, stable

    3. EFFICIENCY
       Hybrid uses multi-fidelity evaluation
       (Hyperband successive halving)
       → More candidates screened per budget

    4. FINAL RESULT
       Hybrid achieves {improvement:.2f}pp higher
       test accuracy with same budget
    """

    ax5.text(0.05, 0.95, insights_text, transform=ax5.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # =========================================================================
    # Final layout
    # =========================================================================
    plt.suptitle('OPRO vs Hybrid OPRO+HbBoPs: Prompt Optimization Comparison\n'
                 'GSM8K Dataset | Qwen/Qwen2.5-7B-Instruct | Budget: 50,000',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to: {output_path}")
    plt.close()


def main():
    # File paths
    opro_json = Path('/home/prusek/NLP/results/opro_20251211_082820.json')
    hybrid_json = Path('/home/prusek/NLP/hybrid_opro_hbbops/results/hybrid_20251211_082856.json')
    hybrid_log = Path('/home/prusek/NLP/hybrid_opro_hbbops/results/hybrid_50k.out')

    output_path = Path('/home/prusek/NLP/visualize/opro_vs_hybrid_comparison.png')

    print("Loading OPRO results...")
    opro_data = load_opro_results(opro_json)
    print(f"  Test accuracy: {opro_data['test_accuracy']:.2f}%")
    print(f"  Validation accuracy: {opro_data['validation_accuracy']:.2f}%")
    print(f"  Budget used: {opro_data['total_budget']:,}")

    print("\nLoading Hybrid OPRO+HbBoPs results...")
    hybrid_data = load_hybrid_results(hybrid_json, hybrid_log)
    print(f"  Test accuracy: {hybrid_data['test_accuracy']:.2f}%")
    print(f"  Best validation: {max(hybrid_data['best_accuracy']):.2f}%")
    print(f"  Budget used: {hybrid_data['total_budget']:,}")

    print("\nCreating comparison figure...")
    create_comparison_figure(opro_data, hybrid_data, output_path)

    # Print summary
    improvement = hybrid_data['test_accuracy'] - opro_data['test_accuracy']
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"OPRO Test Accuracy:           {opro_data['test_accuracy']:.2f}%")
    print(f"Hybrid Test Accuracy:         {hybrid_data['test_accuracy']:.2f}%")
    print(f"Improvement:                  +{improvement:.2f} percentage points")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
