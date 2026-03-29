"""
Policy evaluation on High-Fidelity model.
Validates RL policy or baseline heuristic against the true ODE dynamics.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    HF_PARAMS, EPISODE_HORIZON, REWARD_WEIGHTS, 
    INITIAL_STATE_RANGES, MAX_SUBSTRATE_ADDITION
)
from hf_model import query_single_step, sample_initial_state
from data_builder import calculate_reward
from utils import BudgetTracker, format_time


# ============================================================================
# BASELINE POLICIES
# ============================================================================

def baseline_policy_predict(state: np.ndarray) -> np.ndarray:
    """Simple heuristic baseline: Feed more if substrate is low.
    
    Args:
        state: [biomass, substrate, volume]
    Returns:
        action: [feed_rate]
    """
    substrate = state[1]
    if substrate < 2.0:
        return np.array([MAX_SUBSTRATE_ADDITION * 0.8])
    elif substrate < 10.0:
        return np.array([MAX_SUBSTRATE_ADDITION * 0.3])
    else:
        return np.array([0.0])


# ============================================================================
# EVALUATION CORE
# ============================================================================

def run_hf_episode(
    policy,
    policy_type: str = 'rl',
    initial_state: np.ndarray = None,
    horizon: int = EPISODE_HORIZON,
    params: dict = None,
    budget_tracker: BudgetTracker = None,
    verbose: bool = False
) -> dict:
    """Run a single episode on the High-Fidelity model.
    
    Args:
        policy: RL policy object or callable for baseline
        policy_type: 'rl' or 'baseline'
        initial_state: Starting state
        horizon: Max steps
        params: HF model parameters
        budget_tracker: Tracker for computational budget
        verbose: Print step details
        
    Returns:
        Dictionary with episode metrics
    """
    if params is None:
        params = HF_PARAMS
    if initial_state is None:
        initial_state = sample_initial_state()
        
    current_state = initial_state.copy()
    total_reward = 0.0
    trajectory = []
    
    # Metrics tracking
    biomass_history = [current_state[0]]
    substrate_history = [current_state[1]]
    volume_history = [current_state[2]]
    actions_history = []
    rewards_history = []
    
    for step in range(horizon):
        # 1. Get action
        if policy_type == 'rl':
            action, _ = policy.predict(current_state, deterministic=True)
            action = np.array(action).flatten()
        else:
            action = policy(current_state)
            
        action = np.clip(action, 0.0, MAX_SUBSTRATE_ADDITION)
        
        # 2. Query HF model (Solve ODE)
        next_state, elapsed_time = query_single_step(current_state, action, params)
        
        # Update budget
        if budget_tracker:
            budget_tracker.update_budget(elapsed_time)
            
        # 3. Calculate True Reward
        reward = calculate_reward(current_state, action, next_state)
        
        # 4. Store and Update
        trajectory.append((current_state, action, next_state, reward))
        total_reward += reward
        
        biomass_history.append(next_state[0])
        substrate_history.append(next_state[1])
        volume_history.append(next_state[2])
        actions_history.append(action[0])
        rewards_history.append(reward)
        
        current_state = next_state
        
        if verbose and step % 10 == 0:
            print(f"  Step {step}: X={current_state[0]:.2f}, S={current_state[1]:.2f}, R={reward:.3f}")
            
        # Termination conditions
        if current_state[0] < 1e-6 or current_state[1] < 0:
            if verbose: print(f"  Terminated early at step {step}")
            break
            
    return {
        'total_reward': total_reward,
        'final_biomass': current_state[0],
        'avg_biomass': np.mean(biomass_history),
        'total_feed': np.sum(actions_history),
        'episode_length': len(actions_history),
        'biomass_history': biomass_history,
        'substrate_history': substrate_history,
        'volume_history': volume_history,
        'actions_history': actions_history,
        'rewards_history': rewards_history
    }


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate policy on HF model')
    parser.add_argument('--policy_path', type=str, help='Path to RL policy .zip')
    parser.add_argument('--policy_type', type=str, default='rl', choices=['rl', 'baseline'])
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--use_budget', action='store_true', help='Track HF budget')
    parser.add_argument('--output_file', type=str, default='results/eval_results.json')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"POLICY EVALUATION: {args.policy_type.upper()}")
    print("=" * 70)
    
    # Initialize budget tracker
    budget_tracker = None
    if args.use_budget:
        budget_tracker = BudgetTracker()
        print(f"Budget: {format_time(budget_tracker.get_remaining())} remaining")
        
    # Load policy
    if args.policy_type == 'rl':
        if not args.policy_path:
            raise ValueError("--policy_path is required for RL evaluation")
        print(f"Loading RL policy from: {args.policy_path}")
        policy = PPO.load(args.policy_path)
    else:
        print("Using baseline heuristic policy")
        policy = baseline_policy_predict
        
    all_episode_results = []
    
    for i in range(args.n_episodes):
        if budget_tracker and not budget_tracker.check_budget(0.5): # Estimate 0.5s per episode
            print(f"Budget exhausted before episode {i}")
            break
            
        print(f"Running Episode {i+1}/{args.n_episodes}...")
        result = run_hf_episode(
            policy, 
            args.policy_type, 
            budget_tracker=budget_tracker,
            verbose=args.verbose
        )
        
        # Remove history arrays for JSON summary but keep scalars
        summary = {k: v for k, v in result.items() if not isinstance(v, list)}
        all_episode_results.append(summary)
        
        print(f"  Reward: {result['total_reward']:.3f} | Final Biomass: {result['final_biomass']:.2f}")

    # Calculate aggregate stats
    rewards = [r['total_reward'] for r in all_episode_results]
    stats = {
        'policy_type': args.policy_type,
        'policy_path': args.policy_path,
        'n_episodes': len(all_episode_results),
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'min_reward': float(np.min(rewards)),
        'max_reward': float(np.max(rewards)),
        'episodes': all_episode_results
    }
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETE")
    print(f"Mean Reward: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")
    print(f"Results saved to: {output_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()
