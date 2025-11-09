"""
Data builder for generating initial datasets using exploratory policies.
Implements random exploration with HF model and budget tracking.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    HF_PARAMS, INITIAL_DATASET_EPISODES, MAX_SUBSTRATE_ADDITION,
    get_dataset_path, EPISODE_HORIZON
)
from utils import BudgetTracker, save_dataset, format_time
from hf_model import simulate_episode, sample_initial_state
from logger import ExperimentLogger


# ============================================================================
# EXPLORATORY POLICIES
# ============================================================================

class RandomActionPolicy:
    """Random action policy for exploration."""
    
    def __init__(self, action_dim: int = 1, action_bounds: tuple = (0.0, MAX_SUBSTRATE_ADDITION)):
        self.action_dim = action_dim
        self.action_low, self.action_high = action_bounds
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Sample random action."""
        return np.random.uniform(self.action_low, self.action_high, size=self.action_dim)


class EpsilonGreedyPolicy:
    """Epsilon-greedy policy that sometimes adds more substrate when biomass is low."""
    
    def __init__(self, epsilon: float = 0.3, action_dim: int = 1,
                 action_bounds: tuple = (0.0, MAX_SUBSTRATE_ADDITION)):
        self.epsilon = epsilon
        self.action_dim = action_dim
        self.action_low, self.action_high = action_bounds
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Choose action based on simple heuristic or random."""
        if np.random.rand() < self.epsilon:
            # Random exploration
            return np.random.uniform(self.action_low, self.action_high, size=self.action_dim)
        else:
            # Simple heuristic: add more substrate if substrate is low
            biomass, substrate = state[0], state[1]
            if substrate < 5.0:
                action = np.array([self.action_high * 0.8])  # Add significant substrate
            elif substrate < 10.0:
                action = np.array([self.action_high * 0.4])  # Add moderate substrate
            else:
                action = np.array([self.action_high * 0.1])  # Add little substrate
            return action


# ============================================================================
# REWARD CALCULATION
# ============================================================================

def calculate_reward(state: np.ndarray, action: np.ndarray, next_state: np.ndarray,
                     weights: dict = None) -> float:
    """Calculate reward for a transition.
    
    Multi-objective reward:
    - Reward biomass growth
    - Penalize substrate consumption cost
    
    Args:
        state: Current state [biomass, substrate]
        action: Action taken [substrate_addition]
        next_state: Next state [biomass, substrate]
        weights: Reward weights (default from config)
        
    Returns:
        Reward value
    """
    from config import REWARD_WEIGHTS
    
    if weights is None:
        weights = REWARD_WEIGHTS
    
    # Biomass growth
    biomass_growth = next_state[0] - state[0]
    
    # Substrate cost (action is substrate addition)
    substrate_cost = action[0]
    
    # Combined reward
    reward = (weights['biomass_weight'] * biomass_growth - 
              weights['substrate_cost'] * substrate_cost)
    
    return reward


# ============================================================================
# DATASET GENERATION
# ============================================================================

def generate_dataset(
    n_episodes: int,
    policy,
    budget_tracker: BudgetTracker = None,
    horizon: int = EPISODE_HORIZON,
    logger: ExperimentLogger = None,
    verbose: bool = True
) -> pd.DataFrame:
    """Generate dataset by running HF episodes with given policy.
    
    Args:
        n_episodes: Number of episodes to generate
        policy: Policy function that takes state and returns action
        budget_tracker: Budget tracker (optional)
        horizon: Episode horizon
        logger: Experiment logger (optional)
        verbose: Print progress
        
    Returns:
        DataFrame with columns [state, action, next_state, reward]
    """
    all_transitions = []
    total_time = 0.0
    episodes_completed = 0
    
    if verbose:
        print(f"\nGenerating dataset with {n_episodes} episodes...")
        print(f"Policy: {policy.__class__.__name__}")
        print(f"Horizon: {horizon} steps per episode")
    
    # Estimate time per episode (rough estimate)
    estimated_time_per_episode = 0.5  # seconds (will be updated after first episode)
    
    for episode_idx in range(n_episodes):
        # Check budget before episode
        if budget_tracker is not None:
            if not budget_tracker.check_budget(estimated_time_per_episode):
                print(f"\n⚠ Budget exhausted after {episodes_completed} episodes!")
                break
        
        # Sample initial state
        initial_state = sample_initial_state()
        
        # Run episode
        trajectory, elapsed_time = simulate_episode(
            initial_state,
            policy,
            horizon=horizon,
            params=HF_PARAMS
        )
        
        # Update budget
        total_time += elapsed_time
        if budget_tracker is not None:
            budget_tracker.update_budget(elapsed_time, n_queries=len(trajectory))
        
        # Update time estimate
        if episode_idx == 0:
            estimated_time_per_episode = elapsed_time * 1.2  # Add 20% safety margin
        
        # Calculate rewards for transitions
        for state, action, next_state, _ in trajectory:
            reward = calculate_reward(state, action, next_state)
            all_transitions.append({
                'state': state,
                'action': action,
                'next_state': next_state,
                'reward': reward
            })
        
        episodes_completed += 1
        
        # Progress update
        if verbose and (episode_idx + 1) % 10 == 0:
            avg_time = total_time / (episode_idx + 1)
            remaining = n_episodes - (episode_idx + 1)
            est_remaining_time = avg_time * remaining
            
            print(f"  Episode {episode_idx + 1}/{n_episodes} | "
                  f"Avg time: {avg_time:.2f}s | "
                  f"Est. remaining: {format_time(est_remaining_time)} | "
                  f"Transitions: {len(all_transitions)}")
            
            if budget_tracker is not None:
                remaining_budget = budget_tracker.get_remaining_time()
                print(f"    Budget remaining: {format_time(remaining_budget)} "
                      f"({budget_tracker.get_remaining_percentage():.1f}%)")
        
        # Log to experiment tracker
        if logger is not None:
            logger.log_metrics({
                'data_generation/episode': episode_idx,
                'data_generation/transitions': len(all_transitions),
                'data_generation/episode_time': elapsed_time
            }, step=episode_idx)
    
    # Create DataFrame
    dataset = pd.DataFrame(all_transitions)
    
    # Summary
    if verbose:
        print(f"\n✓ Dataset generation complete!")
        print(f"  Episodes: {episodes_completed}/{n_episodes}")
        print(f"  Total transitions: {len(dataset)}")
        print(f"  Total time: {format_time(total_time)}")
        print(f"  Avg transitions per episode: {len(dataset)/episodes_completed:.1f}")
        
        if budget_tracker is not None:
            summary = budget_tracker.get_summary()
            print(f"  Budget used: {summary['spent_percentage']:.1f}%")
            print(f"  Budget remaining: {format_time(summary['remaining_time_seconds'])}")
    
    return dataset


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate initial dataset using exploratory policy'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Output path for dataset (e.g., data/D_v1.pkl)'
    )
    parser.add_argument(
        '--n_episodes',
        type=int,
        default=INITIAL_DATASET_EPISODES,
        help=f'Number of episodes to generate (default: {INITIAL_DATASET_EPISODES})'
    )
    parser.add_argument(
        '--policy',
        type=str,
        default='random',
        choices=['random', 'epsilon_greedy'],
        help='Exploration policy to use (default: random)'
    )
    parser.add_argument(
        '--horizon',
        type=int,
        default=EPISODE_HORIZON,
        help=f'Episode horizon (default: {EPISODE_HORIZON})'
    )
    parser.add_argument(
        '--use_budget',
        action='store_true',
        help='Use budget tracking (default: False for initial dataset)'
    )
    parser.add_argument(
        '--log_experiment',
        action='store_true',
        help='Log to WandB/TensorBoard'
    )
    
    args = parser.parse_args()
    
    # Initialize budget tracker if requested
    budget_tracker = None
    if args.use_budget:
        budget_tracker = BudgetTracker()
        print(f"\n📊 Budget tracking enabled")
        print(f"   Total budget: {format_time(budget_tracker.total_budget)}")
        print(f"   Remaining: {format_time(budget_tracker.get_remaining_time())}")
    
    # Initialize logger if requested
    logger = None
    if args.log_experiment:
        logger = ExperimentLogger(
            experiment_name='data_generation',
            config={
                'n_episodes': args.n_episodes,
                'policy': args.policy,
                'horizon': args.horizon,
                'use_budget': args.use_budget
            }
        )
    
    # Select policy
    if args.policy == 'random':
        policy = RandomActionPolicy()
    elif args.policy == 'epsilon_greedy':
        policy = EpsilonGreedyPolicy(epsilon=0.3)
    else:
        raise ValueError(f"Unknown policy: {args.policy}")
    
    # Generate dataset
    dataset = generate_dataset(
        n_episodes=args.n_episodes,
        policy=policy,
        budget_tracker=budget_tracker,
        horizon=args.horizon,
        logger=logger,
        verbose=True
    )
    
    # Save dataset
    output_path = Path(args.output_path)
    save_dataset(dataset, output_path)
    
    # Statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"   Shape: {dataset.shape}")
    print(f"   Reward range: [{dataset['reward'].min():.3f}, {dataset['reward'].max():.3f}]")
    print(f"   Mean reward: {dataset['reward'].mean():.3f} ± {dataset['reward'].std():.3f}")
    
    # Cleanup
    if logger is not None:
        logger.finish()
    
    print(f"\n✅ Dataset saved to: {output_path}")


if __name__ == '__main__':
    main()
