"""
Main experiment orchestration: Complete active learning loop.
Coordinates all phases from initialization through iterative improvement.
"""
import sys
import json
import subprocess
from pathlib import Path

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    AL_CONFIG, INITIAL_DATASET_EPISODES, EPISODE_HORIZON,
    get_dataset_path, get_scaler_path, get_surrogate_path, get_policy_path,
    get_results_path
)
from utils import BudgetTracker, format_time
from logger import ExperimentLogger


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_command(cmd: list, description: str, verbose: bool = True):
    """Run a shell command and handle errors.
    
    Args:
        cmd: Command as list of strings
        description: Description of what the command does
        verbose: Print output
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"{description}")
        print(f"{'='*70}")
        print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=not verbose)
    
    if result.returncode != 0:
        print(f"\n❌ Error running: {description}")
        print(f"Return code: {result.returncode}")
        if not verbose:
            print(f"stdout: {result.stdout.decode()}")
            print(f"stderr: {result.stderr.decode()}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    
    if verbose:
        print(f"✓ Completed: {description}\n")
    
    return result


# ============================================================================
# ITERATION 0: INITIALIZATION
# ============================================================================

def run_iteration_zero(logger: ExperimentLogger = None, verbose: bool = True):
    """Run iteration 0: Generate initial dataset and train first models.
    
    Returns:
        Dictionary with iteration 0 results
    """
    print("\n" + "="*70)
    print("ITERATION 0: INITIALIZATION")
    print("="*70)
    
    version = 1
    
    # Step 1: Generate initial dataset
    run_command([
        'python', 'src/data_builder.py',
        '--output_path', str(get_dataset_path(version)),
        '--n_episodes', str(INITIAL_DATASET_EPISODES),
        '--policy', 'random'
    ], "Step 1: Generate initial dataset", verbose)
    
    # Step 2: Fit scalers
    run_command([
        'python', 'src/fit_scaler.py',
        '--dataset_path', str(get_dataset_path(version)),
        '--output_dir', 'models',
        '--version', str(version),
        '--verify'
    ], "Step 2: Fit scalers", verbose)
    
    # Step 3: Train surrogate ensemble
    n_ensemble = AL_CONFIG['n_ensemble_models']
    for i in range(n_ensemble):
        run_command([
            'python', 'src/surrogate_model.py',
            '--dataset_path', str(get_dataset_path(version)),
            '--state_scaler_path', str(get_scaler_path('state', version)),
            '--action_scaler_path', str(get_scaler_path('action', version)),
            '--output_dir', 'models',
            '--version', str(version),
            '--ensemble_index', str(i)
        ], f"Step 3.{i+1}: Train surrogate ensemble member {i}", verbose)
    
    # Step 4: Train RL policy
    run_command([
        'python', 'src/train_rl.py',
        '--surrogate_path', str(get_surrogate_path(version, ensemble_index=0)),
        '--state_scaler_path', str(get_scaler_path('state', version)),
        '--action_scaler_path', str(get_scaler_path('action', version)),
        '--save_path', str(get_policy_path(version))
    ], "Step 4: Train RL policy", verbose)
    
    # Step 5: Evaluate on HF model
    run_command([
        'python', 'src/evaluate_policy.py',
        '--policy_path', str(get_policy_path(version)),
        '--policy_type', 'rl',
        '--n_episodes', '20',
        '--use_budget',
        '--output_path', str(get_results_path(f'eval_v{version}'))
    ], "Step 5: Evaluate policy on HF model", verbose)
    
    # Load evaluation results
    with open(get_results_path(f'eval_v{version}'), 'r') as f:
        eval_results = json.load(f)
    
    # Get budget tracker state
    budget_tracker = BudgetTracker()
    budget_summary = budget_tracker.get_summary()
    
    results = {
        'iteration': 0,
        'version': version,
        'mean_reward': eval_results['mean_reward'],
        'std_reward': eval_results['std_reward'],
        'n_episodes': eval_results['n_episodes'],
        'total_queries': budget_summary['total_queries'],
        'spent_time': budget_summary['spent_time_seconds'],
        'remaining_time': budget_summary['remaining_time_seconds']
    }
    
    # Log to experiment tracker
    if logger:
        logger.log_metrics({
            'iteration': 0,
            'mean_reward': results['mean_reward'],
            'std_reward': results['std_reward'],
            'total_queries': results['total_queries']
        }, step=0)
    
    print(f"\n✓ Iteration 0 complete!")
    print(f"   Mean reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"   Budget used: {budget_summary['spent_percentage']:.1f}%")
    
    return results


# ============================================================================
# ACTIVE LEARNING ITERATIONS
# ============================================================================

def run_al_iteration(iteration: int, logger: ExperimentLogger = None, 
                    verbose: bool = True):
    """Run one active learning iteration.
    
    Args:
        iteration: Iteration number (>= 1)
        logger: Experiment logger
        verbose: Print progress
        
    Returns:
        Dictionary with iteration results
    """
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration}: ACTIVE LEARNING")
    print(f"{'='*70}")
    
    prev_version = iteration
    new_version = iteration + 1
    
    # Check budget before starting
    budget_tracker = BudgetTracker()
    if budget_tracker.get_remaining_percentage() < 5:
        print(f"\n⚠️  Budget nearly exhausted (<5%), stopping iterations")
        return None
    
    # Step 1: Active learning (query selection and HF queries)
    run_command([
        'python', 'src/active_learning.py',
        '--dataset_path', str(get_dataset_path(prev_version)),
        '--policy_path', str(get_policy_path(prev_version)),
        '--ensemble_dir', 'models',
        '--state_scaler_path', str(get_scaler_path('state', prev_version)),
        '--action_scaler_path', str(get_scaler_path('action', prev_version)),
        '--version', str(prev_version),
        '--n_queries', str(AL_CONFIG['n_queries_per_iteration']),
        '--output_path', str(get_dataset_path(new_version)),
        '--selection_method', 'uncertainty'
    ], f"Step 1: Active Learning (query selection & HF queries)", verbose)
    
    # Step 2: Refit scalers on augmented dataset
    run_command([
        'python', 'src/fit_scaler.py',
        '--dataset_path', str(get_dataset_path(new_version)),
        '--output_dir', 'models',
        '--version', str(new_version)
    ], "Step 2: Refit scalers", verbose)
    
    # Step 3: Retrain surrogate ensemble
    n_ensemble = AL_CONFIG['n_ensemble_models']
    for i in range(n_ensemble):
        run_command([
            'python', 'src/surrogate_model.py',
            '--dataset_path', str(get_dataset_path(new_version)),
            '--state_scaler_path', str(get_scaler_path('state', new_version)),
            '--action_scaler_path', str(get_scaler_path('action', new_version)),
            '--output_dir', 'models',
            '--version', str(new_version),
            '--ensemble_index', str(i)
        ], f"Step 3.{i+1}: Retrain surrogate ensemble member {i}", verbose)
    
    # Step 4: Retrain RL policy
    run_command([
        'python', 'src/train_rl.py',
        '--surrogate_path', str(get_surrogate_path(new_version, ensemble_index=0)),
        '--state_scaler_path', str(get_scaler_path('state', new_version)),
        '--action_scaler_path', str(get_scaler_path('action', new_version)),
        '--save_path', str(get_policy_path(new_version))
    ], "Step 4: Retrain RL policy", verbose)
    
    # Step 5: Evaluate on HF model
    run_command([
        'python', 'src/evaluate_policy.py',
        '--policy_path', str(get_policy_path(new_version)),
        '--policy_type', 'rl',
        '--n_episodes', '20',
        '--use_budget',
        '--output_path', str(get_results_path(f'eval_v{new_version}'))
    ], "Step 5: Evaluate policy on HF model", verbose)
    
    # Load evaluation results
    with open(get_results_path(f'eval_v{new_version}'), 'r') as f:
        eval_results = json.load(f)
    
    # Get budget tracker state
    budget_tracker = BudgetTracker()
    budget_summary = budget_tracker.get_summary()
    
    results = {
        'iteration': iteration,
        'version': new_version,
        'mean_reward': eval_results['mean_reward'],
        'std_reward': eval_results['std_reward'],
        'n_episodes': eval_results['n_episodes'],
        'total_queries': budget_summary['total_queries'],
        'spent_time': budget_summary['spent_time_seconds'],
        'remaining_time': budget_summary['remaining_time_seconds']
    }
    
    # Log to experiment tracker
    if logger:
        logger.log_metrics({
            'iteration': iteration,
            'mean_reward': results['mean_reward'],
            'std_reward': results['std_reward'],
            'total_queries': results['total_queries'],
            'budget_remaining_pct': budget_summary['remaining_percentage']
        }, step=iteration)
    
    print(f"\n✓ Iteration {iteration} complete!")
    print(f"   Mean reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"   Budget used: {budget_summary['spent_percentage']:.1f}%")
    print(f"   Budget remaining: {format_time(budget_summary['remaining_time_seconds'])}")
    
    return results


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run complete active learning experiment'
    )
    parser.add_argument(
        '--max_iterations',
        type=int,
        default=AL_CONFIG['max_iterations'],
        help=f"Maximum AL iterations (default: {AL_CONFIG['max_iterations']})"
    )
    parser.add_argument(
        '--log_experiment',
        action='store_true',
        help='Log to WandB/TensorBoard'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed progress'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("MAIN EXPERIMENT: ACTIVE LEARNING FOR FERMENTATION OPTIMIZATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Queries per iteration: {AL_CONFIG['n_queries_per_iteration']}")
    print(f"  Ensemble size: {AL_CONFIG['n_ensemble_models']}")
    
    # Initialize logger
    logger = None
    if args.log_experiment:
        logger = ExperimentLogger(
            experiment_name='main_experiment',
            config={
                'max_iterations': args.max_iterations,
                'n_queries_per_iteration': AL_CONFIG['n_queries_per_iteration'],
                'n_ensemble_models': AL_CONFIG['n_ensemble_models']
            }
        )
    
    # Initialize budget tracker
    budget_tracker = BudgetTracker()
    print(f"\n📊 Initial budget: {format_time(budget_tracker.total_budget)}")
    
    all_results = []
    
    try:
        # Run iteration 0
        results_0 = run_iteration_zero(logger, args.verbose)
        all_results.append(results_0)
        
        # Run active learning iterations
        for iteration in range(1, args.max_iterations + 1):
            # Check budget
            budget_tracker = BudgetTracker()
            if not budget_tracker.check_budget(1000):  # Need at least ~15 min
                print(f"\n⚠️  Insufficient budget for iteration {iteration}, stopping")
                break
            
            results = run_al_iteration(iteration, logger, args.verbose)
            
            if results is None:
                break
            
            all_results.append(results)
        
    except Exception as e:
        print(f"\n❌ Error during experiment: {e}")
        raise
    
    finally:
        # Save final results
        final_results = {
            'iterations': all_results,
            'config': {
                'max_iterations': args.max_iterations,
                'n_queries_per_iteration': AL_CONFIG['n_queries_per_iteration'],
                'n_ensemble_models': AL_CONFIG['n_ensemble_models']
            }
        }
        
        results_path = get_results_path('main_experiment_results')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n💾 Final results saved to: {results_path}")
        
        # Final budget summary
        budget_tracker = BudgetTracker()
        summary = budget_tracker.get_summary()
        
        print(f"\n{'='*70}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*70}")
        print(f"Iterations completed: {len(all_results)}")
        print(f"Final mean reward: {all_results[-1]['mean_reward']:.3f} ± {all_results[-1]['std_reward']:.3f}")
        print(f"Total HF queries: {summary['total_queries']}")
        print(f"Budget used: {summary['spent_percentage']:.1f}%")
        print(f"Budget remaining: {format_time(summary['remaining_time_seconds'])}")
        print(f"{'='*70}")
        
        if logger:
            logger.finish()


if __name__ == '__main__':
    main()
