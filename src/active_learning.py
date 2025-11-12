"""
Active Learning module: uncertainty-based query selection and dataset augmentation.
Implements ensemble uncertainty estimation and HF model querying.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    STATE_DIM, ACTION_DIM, AL_CONFIG,
    get_surrogate_path, get_scaler_path, get_dataset_path
)
from utils import (
    load_dataset, save_dataset, load_scaler,
    BudgetTracker, format_time
)
from surrogate_model import SurrogateModel, get_ensemble_uncertainty
from surrogate_env import SurrogateEnv
from hf_model import query_single_step
from data_builder import calculate_reward


# ============================================================================
# CANDIDATE GENERATION
# ============================================================================

def generate_candidate_queries(
    policy_path: Path,
    surrogate_path: Path,
    state_scaler_path: Path,
    action_scaler_path: Path,
    n_episodes: int = None,
    verbose: bool = True
) -> list:
    """Generate candidate state-action pairs by running policy in surrogate env.
    
    Args:
        policy_path: Path to trained RL policy
        surrogate_path: Path to surrogate model
        state_scaler_path: Path to state scaler
        action_scaler_path: Path to action scaler
        n_episodes: Number of episodes to run
        verbose: Print progress
        
    Returns:
        List of (state, action) tuples
    """
    if n_episodes is None:
        n_episodes = AL_CONFIG['candidate_episodes']
    
    if verbose:
        print(f"\n🎯 Generating candidate queries...")
        print(f"   Running policy for {n_episodes} episodes in surrogate environment")
    
    # Load policy
    policy = PPO.load(policy_path)
    
    # Create surrogate environment
    env = SurrogateEnv(
        surrogate_model_path=surrogate_path,
        state_scaler_path=state_scaler_path,
        action_scaler_path=action_scaler_path
    )
    
    candidates = []
    
    for episode_idx in range(n_episodes):
        obs, info = env.reset()
        
        while True:
            # Get action from policy
            action, _ = policy.predict(obs, deterministic=True)
            action = np.array(action).flatten()
            
            # Store candidate
            candidates.append((obs.copy(), action.copy()))
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
    
    env.close()
    
    if verbose:
        print(f"   ✓ Generated {len(candidates)} candidate queries")
    
    return candidates


# ============================================================================
# QUERY SELECTION
# ============================================================================

def select_best_queries(
    candidates: list,
    ensemble_models: list,
    state_scaler,
    action_scaler,
    n_queries: int,
    selection_method: str = 'uncertainty',
    device: str = 'cpu',
    verbose: bool = True
) -> list:
    """Select best queries based on uncertainty or random sampling.
    
    Args:
        candidates: List of (state, action) tuples
        ensemble_models: List of trained ensemble models
        state_scaler: Fitted state scaler
        action_scaler: Fitted action scaler
        n_queries: Number of queries to select
        selection_method: 'uncertainty' or 'random'
        device: Device for computation
        verbose: Print progress
        
    Returns:
        List of selected (state, action) tuples
    """
    if verbose:
        print(f"\n🔍 Selecting queries using '{selection_method}' method...")
        print(f"   Candidates: {len(candidates)}")
        print(f"   To select: {n_queries}")
    
    if selection_method == 'random':
        # Random selection (for ablation study)
        if verbose:
            print(f"   Random sampling...")
        indices = np.random.choice(len(candidates), size=min(n_queries, len(candidates)), replace=False)
        selected = [candidates[i] for i in indices]
        
    elif selection_method == 'uncertainty':
        # Uncertainty-based selection
        if verbose:
            print(f"   Computing uncertainty for {len(candidates)} candidates...")
            print(f"   Using ensemble of {len(ensemble_models)} models")
        
        uncertainties = []
        
        for idx, (state, action) in enumerate(candidates):
            uncertainty = get_ensemble_uncertainty(
                state, action, ensemble_models,
                state_scaler, action_scaler, device
            )
            uncertainties.append(uncertainty)
            
            if verbose and (idx + 1) % 500 == 0:
                print(f"     Processed {idx + 1}/{len(candidates)} candidates")
        
        uncertainties = np.array(uncertainties)
        
        # Select top-n by uncertainty
        top_indices = np.argsort(uncertainties)[-n_queries:][::-1]
        selected = [candidates[i] for i in top_indices]
        
        if verbose:
            print(f"   ✓ Selected {len(selected)} queries")
            print(f"   Uncertainty range: [{uncertainties.min():.6f}, {uncertainties.max():.6f}]")
            print(f"   Selected mean uncertainty: {uncertainties[top_indices].mean():.6f}")
    
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    
    return selected


# ============================================================================
# HF MODEL QUERYING
# ============================================================================

def query_hf_model(
    queries: list,
    budget_tracker: BudgetTracker,
    params: dict = None,
    verbose: bool = True
) -> list:
    """Query HF model for selected state-action pairs.
    
    Args:
        queries: List of (state, action) tuples
        budget_tracker: Budget tracker
        params: HF model parameters
        verbose: Print progress
        
    Returns:
        List of (state, action, next_state, reward) tuples
    """
    from config import HF_PARAMS
    
    if params is None:
        params = HF_PARAMS
    
    if verbose:
        print(f"\n🔬 Querying HF model...")
        print(f"   Number of queries: {len(queries)}")
    
    # Estimate total time (rough estimate: 0.01s per query)
    estimated_time_per_query = 0.01
    estimated_total_time = len(queries) * estimated_time_per_query
    
    if verbose:
        print(f"   Estimated time: {format_time(estimated_total_time)}")
    
    # Check budget
    if not budget_tracker.check_budget(estimated_total_time):
        # Reduce queries to fit budget
        remaining_time = budget_tracker.get_remaining_time()
        max_queries = int(remaining_time / estimated_time_per_query)
        
        if max_queries < len(queries):
            print(f"   ⚠️  Reducing queries from {len(queries)} to {max_queries} due to budget")
            queries = queries[:max_queries]
        
        if max_queries == 0:
            print(f"   ❌ Insufficient budget for any queries!")
            return []
    
    new_transitions = []
    total_time = 0.0
    
    for idx, (state, action) in enumerate(queries):
        # Query HF model for next state
        next_state, elapsed_time = query_single_step(state, action, params)
        total_time += elapsed_time
        
        # Calculate reward
        reward = calculate_reward(state, action, next_state)
        
        # Store transition
        new_transitions.append({
            'state': state,
            'action': action,
            'next_state': next_state,
            'reward': reward
        })
        
        if verbose and (idx + 1) % 100 == 0:
            print(f"     Queried {idx + 1}/{len(queries)} | "
                  f"Time: {format_time(total_time)}")
    
    # Update budget
    budget_tracker.update_budget(total_time)
    
    if verbose:
        print(f"   ✓ Completed {len(new_transitions)} queries")
        print(f"   Total time: {format_time(total_time)}")
        print(f"   Avg time per query: {total_time/len(queries):.4f}s")
    
    return new_transitions


# ============================================================================
# DATASET UPDATE
# ============================================================================

def augment_dataset(
    old_dataset: pd.DataFrame,
    new_transitions: list,
    verbose: bool = True
) -> pd.DataFrame:
    """Augment dataset with new transitions.
    
    Args:
        old_dataset: Existing dataset
        new_transitions: List of new transition dictionaries
        verbose: Print progress
        
    Returns:
        Augmented dataset
    """
    if verbose:
        print(f"\n📊 Augmenting dataset...")
        print(f"   Old size: {len(old_dataset)}")
        print(f"   New transitions: {len(new_transitions)}")
    
    # Convert new transitions to DataFrame
    new_df = pd.DataFrame(new_transitions)
    
    # Concatenate
    augmented_dataset = pd.concat([old_dataset, new_df], ignore_index=True)
    
    if verbose:
        print(f"   ✓ New size: {len(augmented_dataset)}")
        print(f"   Growth: {len(new_transitions)/len(old_dataset)*100:.1f}%")
    
    return augmented_dataset


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Active learning: select and query uncertain state-action pairs'
    )
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to current dataset')
    parser.add_argument('--policy_path', type=str, required=True,
                       help='Path to trained policy')
    parser.add_argument('--ensemble_dir', type=str, required=True,
                       help='Directory with ensemble models')
    parser.add_argument('--state_scaler_path', type=str, required=True,
                       help='Path to state scaler')
    parser.add_argument('--action_scaler_path', type=str, required=True,
                       help='Path to action scaler')
    parser.add_argument('--version', type=int, required=True,
                       help='Current version number')
    parser.add_argument('--n_queries', type=int, default=None,
                       help='Number of queries to select')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path for augmented dataset')
    parser.add_argument('--selection_method', type=str, default='uncertainty',
                       choices=['uncertainty', 'random'],
                       help='Query selection method')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'])
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ACTIVE LEARNING")
    print("=" * 70)
    
    # Load configuration
    n_queries = args.n_queries or AL_CONFIG['n_queries_per_iteration']
    n_ensemble = AL_CONFIG['n_ensemble_models']
    
    # Initialize budget tracker
    budget_tracker = BudgetTracker(
        budget_file=Path('budget.json'),
        max_budget_seconds=28800  # 8 hours
    )
    print(f"\n📊 Budget status:")
    status = budget_tracker.get_status()
    print(f"   Remaining: {format_time(status['remaining_seconds'])}")
    print(f"   Used: {status['usage_percentage']:.1f}%")
    
    # Load dataset
    print(f"\n📂 Loading dataset from: {args.dataset_path}")
    old_dataset = load_dataset(Path(args.dataset_path))
    
    # Load scalers
    state_scaler = load_scaler(Path(args.state_scaler_path))
    action_scaler = load_scaler(Path(args.action_scaler_path))
    
    # Load ensemble models
    print(f"\n🏗️  Loading ensemble models...")
    ensemble_models = []
    ensemble_dir = Path(args.ensemble_dir)
    
    for i in range(n_ensemble):
        model_path = get_surrogate_path(args.version, ensemble_index=i)
        if not model_path.exists():
            print(f"   ⚠️  Model not found: {model_path}")
            continue
        
        model = SurrogateModel(state_dim=STATE_DIM, action_dim=ACTION_DIM)
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        model.to(args.device)
        model.eval()
        ensemble_models.append(model)
        print(f"   ✓ Loaded: {model_path.name}")
    
    print(f"   Total models loaded: {len(ensemble_models)}")
    
    # Step 1: Generate candidates
    candidates = generate_candidate_queries(
        policy_path=Path(args.policy_path),
        surrogate_path=get_surrogate_path(args.version, ensemble_index=0),
        state_scaler_path=Path(args.state_scaler_path),
        action_scaler_path=Path(args.action_scaler_path)
    )
    
    # Step 2: Select best queries
    selected_queries = select_best_queries(
        candidates=candidates,
        ensemble_models=ensemble_models,
        state_scaler=state_scaler,
        action_scaler=action_scaler,
        n_queries=n_queries,
        selection_method=args.selection_method,
        device=args.device
    )
    
    # Step 3: Query HF model
    new_transitions = query_hf_model(
        queries=selected_queries,
        budget_tracker=budget_tracker
    )
    
    if len(new_transitions) == 0:
        print("\n❌ No new transitions collected (budget exhausted)")
        return
    
    # Step 4: Augment dataset
    augmented_dataset = augment_dataset(old_dataset, new_transitions)
    
    # Save augmented dataset
    output_path = Path(args.output_path)
    save_dataset(augmented_dataset, output_path)
    
    # Final budget status
    print(f"\n📊 Final budget status:")
    status = budget_tracker.get_status()
    print(f"   Used: {status['usage_percentage']:.1f}%")
    print(f"   Remaining: {format_time(status['remaining_seconds'])}")
    print(f"   Total queries: {status['n_queries']}")
    
    print("\n" + "=" * 70)
    print("✅ ACTIVE LEARNING COMPLETE")
    print("=" * 70)
    print(f"Augmented dataset saved: {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
