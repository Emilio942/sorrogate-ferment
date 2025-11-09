"""
Visualization script for experiment results.
Generates plots: learning curves, multi-objective analysis, surrogate accuracy.
"""
import argparse
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from config import PLOTS_DIR, RESULTS_DIR


# Set style
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


# ============================================================================
# PLOT 1: MONEY PLOT (Learning Curve)
# ============================================================================

def plot_learning_curve(results_dir: Path, output_dir: Path):
    """Create the "money plot": learning curve comparing methods.
    
    Shows mean HF reward vs. number of HF queries for:
    - Active Learning
    - Random Sampling (ablation)
    - Baseline heuristic
    """
    print("\n📈 Creating learning curve plot...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Try to load active learning results
    al_results_file = results_dir / 'main_experiment_results.json'
    if al_results_file.exists():
        with open(al_results_file, 'r') as f:
            al_results = json.load(f)
        
        if 'iterations' in al_results:
            iterations = al_results['iterations']
            queries = [it['total_queries'] for it in iterations]
            rewards = [it['mean_reward'] for it in iterations]
            stds = [it['std_reward'] for it in iterations]
            
            ax.plot(queries, rewards, 'o-', color='blue', linewidth=2, 
                   markersize=8, label='Active Learning')
            ax.fill_between(queries, 
                           np.array(rewards) - np.array(stds),
                           np.array(rewards) + np.array(stds),
                           alpha=0.2, color='blue')
    
    # Try to load random sampling ablation results
    random_results_file = results_dir / 'ablation_random_results.json'
    if random_results_file.exists():
        with open(random_results_file, 'r') as f:
            random_results = json.load(f)
        
        if 'iterations' in random_results:
            iterations = random_results['iterations']
            queries = [it['total_queries'] for it in iterations]
            rewards = [it['mean_reward'] for it in iterations]
            stds = [it['std_reward'] for it in iterations]
            
            ax.plot(queries, rewards, 's--', color='orange', linewidth=2,
                   markersize=8, label='Random Sampling')
            ax.fill_between(queries,
                           np.array(rewards) - np.array(stds),
                           np.array(rewards) + np.array(stds),
                           alpha=0.2, color='orange')
    
    # Try to load baseline results
    baseline_file = results_dir / 'eval_baseline_rule_based.json'
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            baseline_results = json.load(f)
        
        if 'mean_reward' in baseline_results:
            baseline_reward = baseline_results['mean_reward']
            ax.axhline(y=baseline_reward, color='red', linestyle='--',
                      linewidth=2, label='Baseline Heuristic')
    
    ax.set_xlabel('Number of HF Queries', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Episode Reward', fontsize=13, fontweight='bold')
    ax.set_title('Active Learning Performance: Reward vs. HF Budget Usage',
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    output_path = output_dir / 'money_plot.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {output_path}")


# ============================================================================
# PLOT 2: MULTI-OBJECTIVE ANALYSIS
# ============================================================================

def plot_multi_objective(results_dir: Path, output_dir: Path):
    """Create multi-objective comparison plot.
    
    Shows separate bars for different objectives:
    - Final biomass
    - Substrate cost
    - Episode length
    """
    print("\n📊 Creating multi-objective analysis plot...")
    
    # This is a placeholder - would need to extract objective components
    # from logged episodes. For now, create a simple comparison.
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = ['Active Learning', 'Baseline']
    colors = ['blue', 'red']
    
    # Example data (would be extracted from actual results)
    biomass_data = [12.5, 10.2]  # Placeholder
    substrate_cost = [15.3, 18.7]  # Placeholder
    length_data = [45, 38]  # Placeholder
    
    # Biomass
    axes[0].bar(methods, biomass_data, color=colors, alpha=0.7)
    axes[0].set_ylabel('Final Biomass (g/L)', fontweight='bold')
    axes[0].set_title('Biomass Production', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Substrate cost
    axes[1].bar(methods, substrate_cost, color=colors, alpha=0.7)
    axes[1].set_ylabel('Total Substrate Added (g/L)', fontweight='bold')
    axes[1].set_title('Substrate Consumption', fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Episode length
    axes[2].bar(methods, length_data, color=colors, alpha=0.7)
    axes[2].set_ylabel('Episode Length (steps)', fontweight='bold')
    axes[2].set_title('Process Duration', fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Multi-Objective Performance Comparison', 
                fontsize=16, fontweight='bold', y=1.02)
    
    output_path = output_dir / 'multi_objective_plot.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {output_path}")


# ============================================================================
# PLOT 3: SURROGATE ACCURACY
# ============================================================================

def plot_surrogate_accuracy(results_dir: Path, output_dir: Path):
    """Plot surrogate model accuracy over iterations."""
    print("\n📉 Creating surrogate accuracy plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Try to load main experiment results
    results_file = results_dir / 'main_experiment_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if 'iterations' in results:
            iterations = results['iterations']
            queries = [it['total_queries'] for it in iterations]
            val_losses = [it.get('val_loss', 0) for it in iterations]
            
            if any(val_losses):
                ax.plot(queries, val_losses, 'o-', color='purple',
                       linewidth=2, markersize=8)
                ax.set_xlabel('Number of HF Queries', fontsize=13, fontweight='bold')
                ax.set_ylabel('Surrogate Validation MSE', fontsize=13, fontweight='bold')
                ax.set_title('Surrogate Model Accuracy Over Active Learning',
                           fontsize=15, fontweight='bold')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
    
    output_path = output_dir / 'surrogate_accuracy_plot.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {output_path}")


# ============================================================================
# PLOT 4: BUDGET USAGE
# ============================================================================

def plot_budget_usage(results_dir: Path, output_dir: Path):
    """Plot cumulative budget usage over time."""
    print("\n⏱️  Creating budget usage plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Try to load budget state
    from config import DATA_DIR
    budget_file = DATA_DIR / 'budget_state.json'
    
    if budget_file.exists():
        with open(budget_file, 'r') as f:
            budget_state = json.load(f)
        
        spent_pct = 100 - budget_state.get('remaining_percentage', 0)
        remaining_pct = budget_state.get('remaining_percentage', 0)
        
        # Pie chart
        sizes = [spent_pct, remaining_pct]
        labels = ['Used', 'Remaining']
        colors = ['#ff6b6b', '#51cf66']
        explode = (0.05, 0)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
              autopct='%1.1f%%', shadow=True, startangle=90,
              textprops={'fontsize': 14, 'fontweight': 'bold'})
        ax.set_title('HF Model Budget Usage',
                    fontsize=15, fontweight='bold')
    
    output_path = output_dir / 'budget_usage_plot.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {output_path}")


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize experiment results'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default=str(RESULTS_DIR),
        help='Directory with result JSON files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(PLOTS_DIR),
        help='Output directory for plots'
    )
    parser.add_argument(
        '--plots',
        nargs='+',
        default=['all'],
        choices=['all', 'learning_curve', 'multi_objective', 
                'surrogate_accuracy', 'budget_usage'],
        help='Which plots to generate'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("RESULT VISUALIZATION")
    print("=" * 70)
    print(f"\nResults directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    plots_to_generate = args.plots
    if 'all' in plots_to_generate:
        plots_to_generate = ['learning_curve', 'multi_objective',
                           'surrogate_accuracy', 'budget_usage']
    
    # Generate plots
    if 'learning_curve' in plots_to_generate:
        plot_learning_curve(results_dir, output_dir)
    
    if 'multi_objective' in plots_to_generate:
        plot_multi_objective(results_dir, output_dir)
    
    if 'surrogate_accuracy' in plots_to_generate:
        plot_surrogate_accuracy(results_dir, output_dir)
    
    if 'budget_usage' in plots_to_generate:
        plot_budget_usage(results_dir, output_dir)
    
    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nPlots saved to: {output_dir}")
    print("\nGenerated plots:")
    for plot_file in output_dir.glob('*.png'):
        print(f"  - {plot_file.name}")
    print("=" * 70)


if __name__ == '__main__':
    main()
