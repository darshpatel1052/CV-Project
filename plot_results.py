"""
Results Plotting & Analysis
============================

Generate publication-quality plots for:
1. mAP comparison bar chart across all experiments
2. Temperature sensitivity curve
3. Loss component ablation bar chart
4. Training convergence curves
5. Per-class AP heatmap
6. Model efficiency scatter (mAP vs FPS vs Parameters)

Usage:
    python plot_results.py --results_dir outputs/metrics
    python plot_results.py --logs_dir logs/
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import re
import glob


# Publication-quality styling
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


DOTA_CLASSES = [
    'plane', 'ship', 'storage-tank', 'baseball-diamond',
    'tennis-court', 'basketball-court', 'ground-track-field',
    'harbor', 'bridge', 'large-vehicle', 'small-vehicle',
    'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool',
    'container-crane'
]


def parse_log_file(log_path):
    """Parse training log to extract loss and mAP values per epoch."""
    losses = []
    maps = []

    with open(log_path, 'r') as f:
        for line in f:
            # Match loss lines: "Epoch X/Y | Loss: 0.1234"
            loss_match = re.search(r'Epoch (\d+)/\d+ \| (?:Avg )?Loss: ([\d.]+)', line)
            if loss_match:
                epoch = int(loss_match.group(1))
                loss = float(loss_match.group(2))
                losses.append((epoch, loss))

            # Match mAP lines: "Val mAP@0.5: 0.1234"
            map_match = re.search(r'Val mAP@0.5: ([\d.]+)', line)
            if map_match:
                mAP = float(map_match.group(1))
                maps.append(mAP)

    return losses, maps


def plot_training_curves(logs_dir, output_dir):
    """Plot training loss convergence for all experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color map for different experiments
    colors = plt.cm.Set2(np.linspace(0, 1, 8))

    experiment_dirs = {
        'Teacher': 'teacher',
        'Student Baseline': 'student_baseline',
        'Student KD': 'student_kd',
    }

    for idx, (name, subdir) in enumerate(experiment_dirs.items()):
        log_files = glob.glob(os.path.join(logs_dir, subdir, '*.log'))
        if not log_files:
            # Try the root
            log_files = glob.glob(os.path.join('.', f'*{subdir}*.log'))

        for log_file in log_files:
            losses, maps = parse_log_file(log_file)
            if losses:
                epochs = [l[0] for l in losses]
                loss_vals = [l[1] for l in losses]
                axes[0].plot(epochs, loss_vals, label=name, color=colors[idx], linewidth=2)

            if maps:
                map_epochs = list(range(1, len(maps) + 1))
                axes[1].plot(map_epochs, maps, 'o-', label=name, color=colors[idx], linewidth=2)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss Convergence')
    axes[0].legend()

    axes[1].set_xlabel('Validation Checkpoint')
    axes[1].set_ylabel('mAP@0.5')
    axes[1].set_title('Validation mAP Progression')
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_convergence.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_temperature_sensitivity(results, output_dir):
    """
    Plot mAP vs Temperature for the temperature sweep experiment.

    Args:
        results: dict mapping T -> mAP value
    """
    if not results:
        print("No temperature sweep results found.")
        return

    temperatures = sorted(results.keys())
    maps = [results[t] for t in temperatures]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(temperatures, maps, 'o-', color='#2196F3', linewidth=2.5,
            markersize=10, markerfacecolor='white', markeredgewidth=2)

    # Highlight the best
    best_idx = np.argmax(maps)
    ax.plot(temperatures[best_idx], maps[best_idx], 'o', color='#FF5722',
            markersize=14, markeredgewidth=3, markerfacecolor='#FF5722', zorder=5)
    ax.annotate(f'Best: T={temperatures[best_idx]}\nmAP={maps[best_idx]:.4f}',
                xy=(temperatures[best_idx], maps[best_idx]),
                xytext=(15, 15), textcoords='offset points',
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#FF5722'))

    ax.set_xlabel('Temperature (T)', fontsize=12)
    ax.set_ylabel('mAP@0.5', fontsize=12)
    ax.set_title('Temperature Sensitivity Analysis\n(α=1.0, β=0.5, γ=1.0)', fontsize=13)
    ax.set_xticks(temperatures)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'temperature_sensitivity.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_loss_ablation(results, output_dir):
    """
    Bar chart comparing loss component ablation results.

    Args:
        results: dict mapping experiment_name -> mAP
    """
    if not results:
        print("No ablation results found.")
        return

    names = list(results.keys())
    maps = list(results.values())

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['#78909C', '#42A5F5', '#66BB6A', '#FFA726'][:len(names)]
    bars = ax.bar(range(len(names)), maps, color=colors, width=0.6,
                  edgecolor='white', linewidth=2)

    # Add value labels
    for bar, val in zip(bars, maps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('mAP@0.5', fontsize=12)
    ax.set_title('Knowledge Distillation Loss Component Ablation', fontsize=13)
    ax.set_ylim(0, max(maps) * 1.15)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'loss_ablation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_model_comparison(results, output_dir):
    """
    Comprehensive bar chart comparing all trained models.

    Args:
        results: list of dicts with 'name', 'mAP', 'fps', 'params'
    """
    if not results:
        print("No model comparison results found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    names = [r['name'] for r in results]
    maps = [r.get('mAP', 0) for r in results]
    fps_vals = [r.get('fps', 0) for r in results]
    params = [r.get('params', 0) for r in results]

    # Color scheme
    colors = ['#1976D2', '#E64A19', '#388E3C', '#7B1FA2', '#F57C00'][:len(names)]

    # mAP comparison
    bars = axes[0].bar(range(len(names)), maps, color=colors, width=0.6)
    for bar, val in zip(bars, maps):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    axes[0].set_title('Detection Performance (mAP@0.5)')
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    axes[0].set_ylabel('mAP@0.5')

    # FPS comparison
    bars = axes[1].bar(range(len(names)), fps_vals, color=colors, width=0.6)
    for bar, val in zip(bars, fps_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{val:.0f}', ha='center', fontsize=9, fontweight='bold')
    axes[1].set_title('Inference Speed (FPS)')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    axes[1].set_ylabel('FPS')

    # Parameters
    bars = axes[2].bar(range(len(names)), [p / 1e6 for p in params], color=colors, width=0.6)
    for bar, val in zip(bars, params):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{val/1e6:.1f}M', ha='center', fontsize=9, fontweight='bold')
    axes[2].set_title('Model Complexity')
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    axes[2].set_ylabel('Parameters (M)')

    plt.suptitle('Model Comparison Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_gamma_sensitivity(results, output_dir):
    """
    Plot mAP vs gamma (feature KD weight) for the gamma sweep.
    """
    if not results:
        print("No gamma sweep results found.")
        return

    gammas = sorted(results.keys())
    maps = [results[g] for g in gammas]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(gammas, maps, 's-', color='#4CAF50', linewidth=2.5,
            markersize=10, markerfacecolor='white', markeredgewidth=2)

    best_idx = np.argmax(maps)
    ax.plot(gammas[best_idx], maps[best_idx], 's', color='#FF5722',
            markersize=14, markeredgewidth=3, markerfacecolor='#FF5722', zorder=5)
    ax.annotate(f'Best: γ={gammas[best_idx]}\nmAP={maps[best_idx]:.4f}',
                xy=(gammas[best_idx], maps[best_idx]),
                xytext=(15, 15), textcoords='offset points',
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#FF5722'))

    ax.set_xlabel('Feature KD Weight (γ)', fontsize=12)
    ax.set_ylabel('mAP@0.5', fontsize=12)
    ax.set_title('Feature KD Weight Sensitivity\n(α=1.0, β=0.5, T=4)', fontsize=13)
    ax.set_xscale('log', base=2)
    ax.set_xticks(gammas)
    ax.set_xticklabels([str(g) for g in gammas])

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'gamma_sensitivity.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate result plots")
    parser.add_argument('--results_dir', type=str, default='outputs/metrics')
    parser.add_argument('--logs_dir', type=str, default='logs')
    parser.add_argument('--output_dir', type=str, default='outputs/plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Generating Result Plots")
    print("=" * 60)

    # 1. Training convergence
    print("\n1. Training convergence curves...")
    plot_training_curves(args.logs_dir, args.output_dir)

    # 2. Try to load experiment results if they exist
    exp_summary_files = glob.glob('outputs/experiments/summary_*.json')
    if exp_summary_files:
        with open(sorted(exp_summary_files)[-1], 'r') as f:
            experiments = json.load(f)
        print(f"\nLoaded {len(experiments)} experiments from summary")

    # Placeholder data for when real results are available
    # Users should replace these with actual checkpoint mAP values
    print("\n2. Temperature sensitivity (populate with real results)...")
    # Example: plot_temperature_sensitivity({2: 0.15, 4: 0.18, 6: 0.17, 8: 0.14}, args.output_dir)

    print("3. Loss ablation (populate with real results)...")
    # Example: plot_loss_ablation({'Baseline': 0.1, 'Logit KD': 0.14, 'Feature KD': 0.16, 'Full KD': 0.18}, args.output_dir)

    print("4. Model comparison (populate with real results)...")
    # Example:
    # plot_model_comparison([
    #     {'name': 'Teacher', 'mAP': 0.35, 'fps': 15, 'params': 30e6},
    #     {'name': 'Student BL', 'mAP': 0.12, 'fps': 120, 'params': 3e6},
    #     {'name': 'Student KD', 'mAP': 0.20, 'fps': 120, 'params': 3e6},
    # ], args.output_dir)

    print(f"\n✅ Plots saved to: {args.output_dir}")
    print("NOTE: Populate with actual results after training experiments.")


if __name__ == '__main__':
    main()
