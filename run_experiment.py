"""
Experiment Runner — Temperature Sweep, Loss Ablation, Gamma Sweep
===================================================================

Automates ablation studies by running multiple KD training configurations
and collecting results. Each experiment modifies specific hyperparameters
while keeping everything else constant.

Usage:
    python run_experiment.py --experiment temperature_sweep --epochs 30
    python run_experiment.py --experiment loss_ablation --epochs 30
    python run_experiment.py --experiment gamma_sweep --epochs 30
    python run_experiment.py --experiment all --epochs 30
"""

import os
import sys
import json
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path
import yaml


def load_config(path='configs/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def run_training(config_path, epochs, subset, tag):
    """Run a single KD training experiment."""
    cmd = [
        sys.executable, 'train_student_kd.py',
        '--config', config_path,
        '--epochs', str(epochs),
    ]
    if subset:
        cmd.extend(['--subset', str(subset)])

    print(f"\n{'='*70}")
    print(f"RUNNING EXPERIMENT: {tag}")
    print(f"Config: {config_path}")
    print(f"Epochs: {epochs}")
    print(f"{'='*70}\n")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start

    return {
        'tag': tag,
        'config': config_path,
        'returncode': result.returncode,
        'elapsed_seconds': elapsed,
        'elapsed_human': f"{elapsed/60:.1f} min",
    }


def create_experiment_config(base_config, modifications, output_path):
    """Create a modified config for a specific experiment."""
    import copy
    config = copy.deepcopy(base_config)

    for key_path, value in modifications.items():
        keys = key_path.split('.')
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value

    save_config(config, output_path)
    return output_path


def temperature_sweep(base_config, args):
    """
    EXP-3A: Temperature Sensitivity Analysis
    Sweep T = [2, 4, 6, 8] with fixed loss weights.
    """
    temperatures = [2.0, 4.0, 6.0, 8.0]
    results = []

    for T in temperatures:
        tag = f"temp_T{T:.0f}"
        config_path = f"configs/experiments/{tag}.yaml"

        create_experiment_config(base_config, {
            'training_student_kd.kd.temperature': T,
            'training_student_kd.kd.alpha': 1.0,
            'training_student_kd.kd.beta': 0.5,
            'training_student_kd.kd.gamma': 1.0,
            'training_student_kd.output_dir': f'./checkpoints/experiments/{tag}',
            'training_student_kd.log_dir': f'./logs/experiments/{tag}',
        }, config_path)

        result = run_training(config_path, args.epochs, args.subset, tag)
        results.append(result)

    return results


def loss_ablation(base_config, args):
    """
    EXP-3B: Loss Component Ablation
    Test each KD component in isolation and combined.
    """
    configs = [
        ('ablation_logit_only',   {'alpha': 1.0, 'beta': 0.5, 'gamma': 0.0}),
        ('ablation_feature_only', {'alpha': 1.0, 'beta': 0.0, 'gamma': 1.0}),
        ('ablation_full_kd',      {'alpha': 1.0, 'beta': 0.5, 'gamma': 1.0}),
    ]

    results = []

    for tag, kd_params in configs:
        config_path = f"configs/experiments/{tag}.yaml"

        create_experiment_config(base_config, {
            'training_student_kd.kd.temperature': 4.0,
            'training_student_kd.kd.alpha': kd_params['alpha'],
            'training_student_kd.kd.beta': kd_params['beta'],
            'training_student_kd.kd.gamma': kd_params['gamma'],
            'training_student_kd.output_dir': f'./checkpoints/experiments/{tag}',
            'training_student_kd.log_dir': f'./logs/experiments/{tag}',
        }, config_path)

        result = run_training(config_path, args.epochs, args.subset, tag)
        results.append(result)

    return results


def gamma_sweep(base_config, args):
    """
    EXP-3C: Feature KD Weight Sensitivity
    Hypothesis: adapters need higher γ due to semantic translation work.
    """
    gammas = [0.25, 0.5, 1.0, 2.0, 4.0]
    results = []

    for gamma in gammas:
        tag = f"gamma_{gamma:.2f}".replace('.', 'p')
        config_path = f"configs/experiments/{tag}.yaml"

        create_experiment_config(base_config, {
            'training_student_kd.kd.temperature': 4.0,
            'training_student_kd.kd.alpha': 1.0,
            'training_student_kd.kd.beta': 0.5,
            'training_student_kd.kd.gamma': gamma,
            'training_student_kd.output_dir': f'./checkpoints/experiments/{tag}',
            'training_student_kd.log_dir': f'./logs/experiments/{tag}',
        }, config_path)

        result = run_training(config_path, args.epochs, args.subset, tag)
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['temperature_sweep', 'loss_ablation', 'gamma_sweep', 'all'],
                        help='Which experiment set to run')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs per experiment')
    parser.add_argument('--subset', type=int, default=None,
                        help='Subset size (None = use config default)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Base config path')
    args = parser.parse_args()

    base_config = load_config(args.config)

    # Create experiment directories
    os.makedirs('configs/experiments', exist_ok=True)
    os.makedirs('outputs/experiments', exist_ok=True)

    all_results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.experiment in ['temperature_sweep', 'all']:
        print("\n" + "=" * 70)
        print("EXPERIMENT SET: Temperature Sensitivity Analysis")
        print("=" * 70)
        results = temperature_sweep(base_config, args)
        all_results.extend(results)

    if args.experiment in ['loss_ablation', 'all']:
        print("\n" + "=" * 70)
        print("EXPERIMENT SET: Loss Component Ablation")
        print("=" * 70)
        results = loss_ablation(base_config, args)
        all_results.extend(results)

    if args.experiment in ['gamma_sweep', 'all']:
        print("\n" + "=" * 70)
        print("EXPERIMENT SET: Feature KD Weight Sensitivity")
        print("=" * 70)
        results = gamma_sweep(base_config, args)
        all_results.extend(results)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"{'Tag':<30} {'Status':<10} {'Time':<15}")
    print("-" * 55)
    for r in all_results:
        status = "✓ OK" if r['returncode'] == 0 else "✗ FAIL"
        print(f"{r['tag']:<30} {status:<10} {r['elapsed_human']:<15}")

    # Save summary
    summary_path = f'outputs/experiments/summary_{timestamp}.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    # Mark successful experiments
    success = sum(1 for r in all_results if r['returncode'] == 0)
    print(f"\n{success}/{len(all_results)} experiments completed successfully")

    if success < len(all_results):
        print("\n⚠ Some experiments failed. Check logs for details.")
        sys.exit(1)


if __name__ == '__main__':
    main()
