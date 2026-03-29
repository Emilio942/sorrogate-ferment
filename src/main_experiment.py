"""
Main experiment orchestration script for SURROGATE-FERMENT.
Runs the complete active learning loop:
1. Initial data generation & surrogate training
2. Iterative RL training & active learning refinement
3. Final evaluation and logging
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    AL_CONFIG, INITIAL_DATASET_EPISODES, ENSEMBLE_SIZE,
    get_dataset_path, get_scaler_path, get_surrogate_path, get_policy_path,
    MAX_BUDGET_SECONDS, BUDGET_STATE_FILE
)
from utils import BudgetTracker, format_time


def run_command(command: list, description: str):
    """Run a shell command and handle errors.
    
    Args:
        command: List of command arguments
        description: Description of the task
    """
    print(f"\n--- {description} ---")
    print(f"Running: {' '.join(command)}")
    
    start_time = time.time()
    result = subprocess.run(command, capture_output=False, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"❌ Error: {description} failed with return code {result.returncode}")
        sys.exit(1)
    
    print(f"✓ Completed in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description='Run complete active learning experiment')
    parser.add_argument('--n_iterations', type=int, default=AL_CONFIG['n_iterations'],
                       help='Number of AL iterations')
    parser.add_argument('--initial_episodes', type=int, default=INITIAL_DATASET_EPISODES,
                       help='Number of initial exploration episodes')
    parser.add_argument('--force_restart', action='store_true', help='Reset budget and start fresh')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 STARTING SURROGATE-FERMENT MAIN EXPERIMENT")
    print("=" * 80)
    
    # 0. Initialize Budget Tracker
    budget_tracker = BudgetTracker(BUDGET_STATE_FILE, MAX_BUDGET_SECONDS)
    if args.force_restart:
        print("Resetting budget tracker...")
        budget_tracker.reset()
    
    status = budget_tracker.get_status()
    print(f"Current Budget Usage: {status['usage_percentage']:.1f}% "
          f"({format_time(status['spent_seconds'])} / {format_time(MAX_BUDGET_SECONDS)})")

    # 1. INITIALIZATION (Iteration 0)
    print("\n" + "#" * 40)
    print("### ITERATION 0: INITIALIZATION ###")
    print("#" * 40)
    
    v0_dataset = get_dataset_path(0, "initial")
    v0_state_scaler = get_scaler_path('state', 0)
    v0_action_scaler = get_scaler_path('action', 0)
    v0_surrogate = get_surrogate_path(0) # This will be the base path for ensemble
    v0_policy = get_policy_path(0)
    
    # 1.1 Data Builder
    run_command([
        "python", "src/data_builder.py",
        "--output_path", str(v0_dataset),
        "--n_episodes", str(args.initial_episodes),
        "--policy", "random",
        "--use_budget"
    ], "Generating initial dataset")
    
    # 1.2 Fit Scalers
    run_command([
        "python", "src/fit_scaler.py",
        "--dataset_path", str(v0_dataset),
        "--version", "0",
        "--verify"
    ], "Fitting initial scalers")
    
    # 1.3 Train Ensemble
    for i in range(ENSEMBLE_SIZE):
        run_command([
            "python", "src/surrogate_model.py",
            "--dataset_path", str(v0_dataset),
            "--state_scaler_path", str(v0_state_scaler),
            "--action_scaler_path", str(v0_action_scaler),
            "--version", "0",
            "--ensemble_index", str(i),
            "--device", args.device
        ], f"Training surrogate ensemble member {i}")
        
    # 1.4 Train RL Policy (Now using Ensemble implicitly)
    run_command([
        "python", "src/train_rl.py",
        "--surrogate_path", str(v0_surrogate.parent), 
        "--state_scaler_path", str(v0_state_scaler),
        "--action_scaler_path", str(v0_action_scaler),
        "--save_path", str(v0_policy),
        "--total_timesteps", str(50000), # Shorter training for iteration 0
        "--evaluate"
    ], "Training initial RL policy on Surrogate Ensemble")
    
    # 1.5 Initial Evaluation
    run_command([
        "python", "src/evaluate_policy.py",
        "--policy_path", str(v0_policy),
        "--n_episodes", "10",
        "--output_file", "results/eval_v0.json"
    ], "Evaluating initial policy on HF model")

    # 2. ITERATIVE LOOP
    current_dataset = v0_dataset
    
    for iteration in range(1, args.n_iterations + 1):
        print("\n" + "#" * 40)
        print(f"### ITERATION {iteration} / {args.n_iterations} ###")
        print("#" * 40)
        
        # Check budget
        status = budget_tracker.get_status()
        if status['remaining_seconds'] < 300: # Less than 5 mins remaining
            print("⚠ Budget exhausted! Terminating experiment.")
            break
            
        prev_v = iteration - 1
        curr_v = iteration
        
        prev_dataset = get_dataset_path(prev_v, "initial" if prev_v == 0 else "augmented")
        prev_policy = get_policy_path(prev_v)
        prev_state_scaler = get_scaler_path('state', prev_v)
        prev_action_scaler = get_scaler_path('action', prev_v)
        
        curr_dataset = get_dataset_path(curr_v, "augmented")
        curr_state_scaler = get_scaler_path('state', curr_v)
        curr_action_scaler = get_scaler_path('action', curr_v)
        curr_surrogate = get_surrogate_path(curr_v)
        curr_policy = get_policy_path(curr_v)
        
        # 2.1 Active Learning Step
        run_command([
            "python", "src/active_learning.py",
            "--dataset_path", str(prev_dataset),
            "--policy_path", str(prev_policy),
            "--ensemble_dir", str(prev_state_scaler.parent),
            "--state_scaler_path", str(prev_state_scaler),
            "--action_scaler_path", str(prev_action_scaler),
            "--version", str(prev_v),
            "--n_queries", str(AL_CONFIG['n_queries_per_iteration']),
            "--output_path", str(curr_dataset),
            "--selection_method", "uncertainty"
        ], f"Active Learning iteration {iteration}")
        
        # 2.2 Refit Scalers
        run_command([
            "python", "src/fit_scaler.py",
            "--dataset_path", str(curr_dataset),
            "--version", str(curr_v)
        ], f"Refitting scalers for version {curr_v}")
        
        # 2.3 Retrain Ensemble
        for i in range(ENSEMBLE_SIZE):
            run_command([
                "python", "src/surrogate_model.py",
                "--dataset_path", str(curr_dataset),
                "--state_scaler_path", str(curr_state_scaler),
                "--action_scaler_path", str(curr_action_scaler),
                "--version", str(curr_v),
                "--ensemble_index", str(i),
                "--device", args.device
            ], f"Retraining surrogate ensemble member {i}")
            
        # 2.4 Retrain RL Policy
        run_command([
            "python", "src/train_rl.py",
            "--surrogate_path", str(curr_surrogate.parent),
            "--state_scaler_path", str(curr_state_scaler),
            "--action_scaler_path", str(curr_action_scaler),
            "--save_path", str(curr_policy),
            "--total_timesteps", str(80000)
        ], f"Retraining RL policy for version {curr_v} on Ensemble")
        
        # 2.5 Evaluate Policy
        run_command([
            "python", "src/evaluate_policy.py",
            "--policy_path", str(curr_policy),
            "--n_episodes", "10",
            "--output_file", f"results/eval_v{curr_v}.json"
        ], f"Evaluating policy v{curr_v} on HF model")

    print("\n" + "=" * 80)
    print("✅ EXPERIMENT COMPLETE")
    print("=" * 80)
    status = budget_tracker.get_status()
    print(f"Final Budget Usage: {status['usage_percentage']:.1f}%")
    print(f"Total HF Queries: {status['n_queries']}")
    print("=" * 80)


if __name__ == '__main__':
    main()
