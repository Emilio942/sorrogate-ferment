"""
Policy evaluation on high-fidelity model.
Evaluates trained RL policies or baseline heuristics on the true dynamics.
"""
import argparse
import sys
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    HF_PARAMS, EPISODE_HORIZON, MAX_SUBSTRATE_ADDITION,
    REWARD_WEIGHTS, get_results_path
)
from utils import BudgetTracker, format_time
from hf_model import simulate_episode, sample_initial_state, query_single_step
from data_builder import calculate_reward


# ============================================================================
# BASELINE POLICIES
# ============================================================================

class BaselinePolicy:
    """Base class for baseline policies."""
    
    def predict(self, observation, deterministic=True):
        """Predict action (compatible with SB3 interface)."""
        action = self._get_action(observation)
        return action, None
    
    def _get_action(self, observation):
        """Get action from observation."""
        raise NotImplementedError


class StaticPolicy(BaselinePolicy):
    """Static policy with constant action."""
    
    def __init__(self, action_value: float = 0.5):
        self.action_value = action_value
    
    def _get_action(self, observation):
        return np.array([self.action_value])


class RuleBasedPolicy(BaselinePolicy):
    """Rule-based policy: add substrate when substrate is low."""
    
    def __init__(self):
        pass
    
    def _get_action(self, observation):
        biomass, substrate = observation[0], observation[1]
        
        # Simple heuristic
        if substrate < 5.0:
            action = np.array([MAX_SUBSTRATE_ADDITION * 0.8])
        elif substrate < 10.0:
            action = np.array([MAX_SUBSTRATE_ADDITION * 0.4])
        else:
            action = np.array([MAX_SUBSTRATE_ADDITION * 0.1])
        
        return action


class RandomPolicy(BaselinePolicy):
    """Random policy."""
    
    def _get_action(self, observation):
        return np.array([np.random.uniform(0, MAX_SUBSTRATE_ADDITION)])


# ============================================================================
# EPISODE EXECUTION ON HF MODEL
# ============================================================================

def run_hf_episode(
    policy,
    initial_state: np.ndarray,
    horizon: int = EPISODE_HORIZON,
    params: dict = None,
    budget_tracker: BudgetTracker = None,
    verbose: bool = False
) -> dict:
    """Run one episode on HF model with given policy.
    
    Args:
        policy: Policy object with predict() method
        initial_state: Initial state
        horizon: Episode horizon
        params: HF model parameters
        budget_tracker: Budget tracker (optional)
        verbose: Print step-by-step info
        
    Returns:
        Dictionary with episode statistics
    """
    import time
    
    if params is None:
        params = HF_PARAMS
    
    current_state = initial_state.copy()
    episode_reward = 0.0
    episode_length = 0
    start_time = time.time()
    
    states = [current_state.copy()]
    actions = []
    rewards = []
    
    for step in range(horizon):
        # Get action from policy
        action, _ = policy.predict(current_state, deterministic=True)
        action = np.array(action).flatten()
        action = np.clip(action, 0.0, MAX_SUBSTRATE_ADDITION)
        
        # Query HF model for next state
        next_state, step_time = query_single_step(current_state, action, params)
        
        # Calculate reward
        reward = calculate_reward(current_state, action, next_state, REWARD_WEIGHTS)
        
        # Store
        states.append(next_state.copy())
        actions.append(action.copy())
        rewards.append(reward)
        
        episode_reward += reward
        episode_length += 1
        
        if verbose:
            print(f"    Step {step+1}: state={next_state}, action={action}, reward={reward:.3f}")
        
        # Early termination
        if next_state[0] < 1e-6 or next_state[1] < 0:
            break
        
        current_state = next_state
    
    elapsed_time = time.time() - start_time
    
    # Update budget if provided
    if budget_tracker is not None:
        budget_tracker.update_budget(elapsed_time, n_queries=episode_length)
    
    return {
        'reward': episode_reward,
        'length': episode_length,
        'time': elapsed_time,
        'final_state': current_state,
        'states': states,
        'actions': actions,
        'rewards': rewards
    }


# ============================================================================
# POLICY EVALUATION
# ============================================================================

def evaluate_policy_on_hf(
    policy,
    policy_type: str,
    n_episodes: int,
    horizon: int = EPISODE_HORIZON,
    params: dict = None,
    budget_tracker: BudgetTracker = None,
    param_shift: dict = None,
    verbose: bool = True
) -> dict:
    """Evaluate policy on HF model over multiple episodes.
    
    Args:
        policy: Policy to evaluate
        policy_type: Type of policy ('rl' or 'baseline')
        n_episodes: Number of episodes
        horizon: Episode horizon
        params: HF model parameters
        budget_tracker: Budget tracker (optional)
        param_shift: Parameter shift for robustness test (optional)
        verbose: Print progress
        
    Returns:
        Dictionary with evaluation statistics
    """
    if params is None:
        params = HF_PARAMS.copy()
    
    # Apply parameter shift if specified
    if param_shift is not None:
        print(f"\n⚠️  Applying parameter shift: {param_shift}")
        for param_name, factor in param_shift.items():
            if param_name in params:
                original = params[param_name]
                params[param_name] = original * factor
                print(f"   {param_name}: {original} → {params[param_name]}")
    
    if verbose:
        print(f"\n📊 Evaluating {policy_type} policy on HF model...")
        print(f"   Episodes: {n_episodes}")
        print(f"   Horizon: {horizon}")
        if budget_tracker is not None:
            print(f"   Budget tracking: enabled")
    
    episode_rewards = []
    episode_lengths = []
    episode_times = []
    episodes_completed = 0
    
    # Estimate time per episode
    estimated_time_per_episode = 1.0  # seconds
    
    for episode_idx in range(n_episodes):
        # Check budget
        if budget_tracker is not None:
            if not budget_tracker.check_budget(estimated_time_per_episode):
                print(f"\n⚠️  Budget exhausted after {episodes_completed} episodes")
                break
        
        # Sample initial state
        initial_state = sample_initial_state()
        
        # Run episode
        result = run_hf_episode(
            policy=policy,
            initial_state=initial_state,
            horizon=horizon,
            params=params,
            budget_tracker=budget_tracker,
            verbose=False
        )
        
        episode_rewards.append(result['reward'])
        episode_lengths.append(result['length'])
        episode_times.append(result['time'])
        episodes_completed += 1
        
        # Update time estimate
        if episode_idx == 0:
            estimated_time_per_episode = result['time'] * 1.2
        
        # Progress
        if verbose and (episode_idx + 1) % 5 == 0:
            current_mean = np.mean(episode_rewards)
            current_std = np.std(episode_rewards)
            print(f"   Episode {episode_idx+1}/{n_episodes} | "
                  f"Reward: {result['reward']:.3f} | "
                  f"Mean so far: {current_mean:.3f} ± {current_std:.3f}")
            
            if budget_tracker is not None:
                remaining = budget_tracker.get_remaining_time()
                print(f"     Budget remaining: {format_time(remaining)}")
    
    # Calculate statistics
    stats = {
        'n_episodes': episodes_completed,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'total_time': np.sum(episode_times),
        'mean_time_per_episode': np.mean(episode_times),
        'policy_type': policy_type,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }
    
    if verbose:
        print(f"\n✓ Evaluation complete!")
        print(f"   Episodes: {stats['n_episodes']}")
        print(f"   Mean reward: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")
        print(f"   Reward range: [{stats['min_reward']:.3f}, {stats['max_reward']:.3f}]")
        print(f"   Mean length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
        print(f"   Total time: {format_time(stats['total_time'])}")
    
    return stats


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate policy on high-fidelity model'
    )
    parser.add_argument(
        '--policy_path',
        type=str,
        default=None,
        help='Path to trained RL policy (required if policy_type=rl)'
    )
    parser.add_argument(
        '--policy_type',
        type=str,
        required=True,
        choices=['rl', 'baseline'],
        help='Type of policy to evaluate'
    )
    parser.add_argument(
        '--baseline_type',
        type=str,
        default='rule_based',
        choices=['static', 'rule_based', 'random'],
        help='Type of baseline policy (if policy_type=baseline)'
    )
    parser.add_argument(
        '--n_episodes',
        type=int,
        default=20,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--use_budget',
        action='store_true',
        help='Use budget tracking'
    )
    parser.add_argument(
        '--param_shift',
        type=str,
        default=None,
        help='Parameter shift for robustness test (e.g., "MU_MAX=1.2")'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Output path for results JSON'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("POLICY EVALUATION ON HF MODEL")
    print("=" * 70)
    
    # Load policy
    if args.policy_type == 'rl':
        if args.policy_path is None:
            raise ValueError("--policy_path required for RL policy evaluation")
        
        print(f"\n📂 Loading RL policy from: {args.policy_path}")
        policy = PPO.load(args.policy_path)
        print(f"   ✓ Policy loaded")
    
    elif args.policy_type == 'baseline':
        print(f"\n🎯 Creating baseline policy: {args.baseline_type}")
        if args.baseline_type == 'static':
            policy = StaticPolicy(action_value=0.5)
        elif args.baseline_type == 'rule_based':
            policy = RuleBasedPolicy()
        elif args.baseline_type == 'random':
            policy = RandomPolicy()
        else:
            raise ValueError(f"Unknown baseline type: {args.baseline_type}")
        print(f"   ✓ Baseline policy created")
    
    # Initialize budget tracker if requested
    budget_tracker = None
    if args.use_budget:
        budget_tracker = BudgetTracker()
        print(f"\n📊 Budget tracking enabled")
        print(f"   Remaining: {format_time(budget_tracker.get_remaining_time())}")
    
    # Parse parameter shift
    param_shift = None
    if args.param_shift:
        parts = args.param_shift.split('=')
        if len(parts) == 2:
            param_name = parts[0].strip()
            factor = float(parts[1].strip())
            param_shift = {param_name: factor}
    
    # Evaluate policy
    stats = evaluate_policy_on_hf(
        policy=policy,
        policy_type=args.policy_type,
        n_episodes=args.n_episodes,
        budget_tracker=budget_tracker,
        param_shift=param_shift,
        verbose=True
    )
    
    # Save results
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        suffix = f"_{args.baseline_type}" if args.policy_type == 'baseline' else ""
        output_path = get_results_path(f"eval_{args.policy_type}{suffix}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python for JSON serialization
    stats_serializable = {
        k: (v.tolist() if isinstance(v, np.ndarray) else 
            float(v) if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in stats.items()
    }
    
    with open(output_path, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    # Budget summary
    if budget_tracker is not None:
        summary = budget_tracker.get_summary()
        print(f"\n📊 Budget Summary:")
        print(f"   Used: {summary['spent_percentage']:.1f}%")
        print(f"   Remaining: {format_time(summary['remaining_time_seconds'])}")
        print(f"   Total queries: {summary['total_queries']}")
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
