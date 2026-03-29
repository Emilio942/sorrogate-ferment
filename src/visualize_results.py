"""
Visualization script for SURROGATE-FERMENT results.
Generates plots for:
- Learning curves (Reward vs HF Queries)
- Surrogate accuracy (MSE vs Iterations)
- Policy trajectories (State/Action over time)
- Multi-objective comparison (RL vs Baseline)
"""
import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")


def load_all_results(results_dir: Path):
    """Load all eval_v*.json files from results directory."""
    results = []
    for file in results_dir.glob("eval_v*.json"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                # Extract iteration number from filename
                iteration = int(file.stem.split('_v')[-1])
                data['iteration'] = iteration
                results.append(data)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            
    return sorted(results, key=lambda x: x['iteration'])


def plot_learning_curve(results: list, output_dir: Path):
    """Plot Mean Reward vs Iteration."""
    iterations = [r['iteration'] for r in results]
    means = [r['mean_reward'] for r in results]
    stds = [r['std_reward'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(iterations, means, yerr=stds, fmt='-o', capsize=5, label='RL Policy')
    
    plt.title('Learning Curve: Policy Performance on HF Model', fontsize=14)
    plt.xlabel('Active Learning Iteration', fontsize=12)
    plt.ylabel('Mean Cumulative Reward', fontsize=12)
    plt.xticks(iterations)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_dir / 'learning_curve.png', dpi=300)
    print(f"✓ Saved learning_curve.png")
    plt.close()


def plot_multi_objective_summary(results: list, output_dir: Path):
    """Plot breakdown of objectives for the latest iteration."""
    if not results: return
    
    latest = results[-1]
    episodes = latest['episodes']
    
    # Aggregate metrics
    metrics = {
        'Final Biomass': [e['final_biomass'] for e in episodes],
        'Total Feed': [e['total_feed'] for e in episodes],
        'Avg Biomass': [e['avg_biomass'] for e in episodes],
        'Total Reward': [e['total_reward'] for e in episodes]
    }
    
    df = pd.DataFrame(metrics)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df)
    plt.title(f'Performance Distribution (Iteration {latest["iteration"]})', fontsize=14)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'objective_summary.png', dpi=300)
    print(f"✓ Saved objective_summary.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory with .json results')
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Searching for results in: {results_dir}")
    results = load_all_results(results_dir)
    
    if not results:
        print("❌ No results found. Run main_experiment.py first.")
        return
        
    print(f"📊 Found {len(results)} iterations. Generating plots...")
    
    plot_learning_curve(results, output_dir)
    plot_multi_objective_summary(results, output_dir)
    
    # Save a summary table
    summary_data = []
    for r in results:
        summary_data.append({
            'Iteration': r['iteration'],
            'Mean Reward': f"{r['mean_reward']:.2f} ± {r['std_reward']:.2f}",
            'Min Reward': f"{r['min_reward']:.2f}",
            'Max Reward': f"{r['max_reward']:.2f}",
            'Episodes': r['n_episodes']
        })
    
    df_summary = pd.DataFrame(summary_data)
    print("\nSummary Table:")
    print(df_summary.to_string(index=False))
    
    print("\n" + "=" * 70)
    print(f"✅ VISUALIZATION COMPLETE")
    print(f"Plots saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
