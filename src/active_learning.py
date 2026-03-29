"""
Active Learning module: uncertainty-based query selection and dataset augmentation.
Implements Expected Fisher Information Gain (EFIG) for optimal candidate selection.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from scipy.optimize import root

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    STATE_DIM, ACTION_DIM, AL_CONFIG, HF_PARAMS, RL_CONFIG,
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
    policy: PPO,
    surrogate_path: Path,
    state_scaler_path: Path,
    action_scaler_path: Path,
    n_episodes: int = None,
    verbose: bool = True
) -> list:
    """Generate candidate state-action pairs using a mixed distribution.
    
    Mixed Distribution:
    - 70% Policy-guided: Transitions visited by the current RL policy.
    - 30% Uniform: Randomly sampled states and actions from the physical range.
    
    Args:
        policy: Trained SB3 PPO policy
        surrogate_path: Path to surrogate model
        state_scaler_path: Path to state scaler
        action_scaler_path: Path to action scaler
        n_episodes: Number of episodes to run for policy-guided part
        verbose: Print progress
        
    Returns:
        List of (state, action) tuples
    """
    from config import INITIAL_STATE_RANGES, MAX_SUBSTRATE_ADDITION
    
    if n_episodes is None:
        n_episodes = AL_CONFIG.get('candidate_episodes', 20)
    
    if verbose:
        print(f"\n🎯 Generating candidate queries (Mixed Distribution)...")
    
    # --- 1. Policy-guided Candidates (70%) ---
    if verbose: print(f"   Running policy for {n_episodes} episodes in surrogate environment...")
    
    env = SurrogateEnv(
        surrogate_model_path=surrogate_path,
        state_scaler_path=state_scaler_path,
        action_scaler_path=action_scaler_path
    )
    
    policy_candidates = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        while True:
            action, _ = policy.predict(obs, deterministic=True)
            action = np.array(action).flatten()
            policy_candidates.append((obs.copy(), action.copy()))
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated: break
    env.close()
    
    # --- 2. Uniform Candidates (30%) ---
    # We want roughly 30% of total candidates to be uniform
    n_uniform = int(len(policy_candidates) * (0.3 / 0.7))
    if verbose: print(f"   Sampling {n_uniform} uniform candidates from physical ranges...")
    
    uniform_candidates = []
    for _ in range(n_uniform):
        # Sample state from ranges
        biomass = np.random.uniform(*INITIAL_STATE_RANGES['biomass'])
        substrate = np.random.uniform(*INITIAL_STATE_RANGES['substrate'])
        volume = np.random.uniform(*INITIAL_STATE_RANGES['volume'])
        state = np.array([biomass, substrate, volume], dtype=np.float32)
        
        # Sample action from range
        action = np.array([np.random.uniform(0.0, MAX_SUBSTRATE_ADDITION)], dtype=np.float32)
        
        uniform_candidates.append((state, action))
        
    candidates = policy_candidates + uniform_candidates
    
    if verbose:
        print(f"   ✓ Total candidates: {len(candidates)} ({len(policy_candidates)} policy, {len(uniform_candidates)} uniform)")
    
    return candidates


# ============================================================================
# EXPECTED FISHER INFORMATION GAIN
# ============================================================================

def get_expected_fisher_information_gain(
    state: np.ndarray,
    action: np.ndarray,
    policy: PPO,
    ensemble_models: list,
    state_scaler,
    action_scaler,
    device: str = 'cpu',
    gamma: float = 0.99
) -> float:
    r"""Calculate the Expected Fisher Information Gain for a given state-action pair.
    
    IG(s,a) = || \nabla_phi \log \pi(a|s) ||_2 * | A(s,a) |
    
    This ensures we only query states that are BOTH:
    1. Highly sensitive for the policy (large score gradient).
    2. Highly relevant to the value landscape (large advantage).
    """
    obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    act_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
    
    # 1. Compute Policy Score Gradient Norm
    dist = policy.policy.get_distribution(obs_tensor)
    log_prob = dist.log_prob(act_tensor)
    
    # Differentiate log_prob w.r.t the action network weights
    actor_params = list(policy.policy.action_net.parameters())
    grads = torch.autograd.grad(log_prob.mean(), actor_params, retain_graph=True)
    grad_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads)).item()
    
    # 2. Compute Advantage Estimate using Ensemble and Semi-Implicit Euler
    with torch.no_grad():
        v_s = policy.policy.predict_values(obs_tensor).item()
        
    state_scaled = state_scaler.transform(state.reshape(1, -1)).flatten()
    action_scaled = action_scaler.transform(action.reshape(1, -1)).flatten()
    
    dt = HF_PARAMS['DT']
    a_tensor = torch.FloatTensor(action_scaled).unsqueeze(0).to(device)
    
    # Semi-Implicit Euler via Root Finding (Matching Phase 2 Env Logic)
    def residual(s_guess_np):
        s_tensor = torch.FloatTensor(s_guess_np).unsqueeze(0).to(device)
        input_tensor = torch.cat([s_tensor, a_tensor], dim=-1)
        with torch.no_grad():
            preds = [model(input_tensor) for model in ensemble_models]
            mean_ds_dt = torch.mean(torch.stack(preds), dim=0).cpu().numpy().flatten()
        return s_guess_np - state_scaled - dt * mean_ds_dt

    # Solve for s_next
    sol = root(residual, state_scaled, method='hybr')
    s_next_scaled = sol.x
    
    next_state = state_scaler.inverse_transform(s_next_scaled.reshape(1, -1)).flatten().astype(np.float32)
    next_state = np.clip(next_state, 0.0, None)
    
    # Get V(s_next)
    obs_next_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
    with torch.no_grad():
        v_s_next = policy.policy.predict_values(obs_next_tensor).item()
        
    # Reward
    reward = calculate_reward(state, action, next_state)
    
    # Advantage
    advantage = reward + gamma * v_s_next - v_s
    
    # 3. Final EFIG Calculation
    efig = grad_norm * abs(advantage)
    
    return float(efig)


# ============================================================================
# QUERY SELECTION
# ============================================================================

def select_best_queries(
    candidates: list,
    ensemble_models: list,
    policy: PPO,
    state_scaler,
    action_scaler,
    n_queries: int,
    selection_method: str = 'fisher_information',
    device: str = 'cpu',
    verbose: bool = True
) -> list:
    """Select best queries based on Fisher Information Gain, uncertainty, or random sampling.
    """
    if verbose:
        print(f"\n🔍 Selecting queries using '{selection_method}' method...")
        print(f"   Candidates: {len(candidates)}")
        print(f"   To select: {n_queries}")
    
    if selection_method == 'random':
        indices = np.random.choice(len(candidates), size=min(n_queries, len(candidates)), replace=False)
        selected = [candidates[i] for i in indices]
        
    elif selection_method == 'uncertainty':
        uncertainties = []
        for idx, (state, action) in enumerate(candidates):
            unc = get_ensemble_uncertainty(state, action, ensemble_models, state_scaler, action_scaler, device)
            uncertainties.append(unc)
            if verbose and (idx + 1) % 500 == 0:
                print(f"     Processed {idx + 1}/{len(candidates)} candidates")
                
        uncertainties = np.array(uncertainties)
        top_indices = np.argsort(uncertainties)[-n_queries:][::-1]
        selected = [candidates[i] for i in top_indices]
        
    elif selection_method == 'fisher_information':
        if verbose:
            print(f"   Computing Expected Fisher Information Gain for {len(candidates)} candidates...")
            
        gamma = RL_CONFIG.get('gamma', 0.99)
        efig_scores = []
        
        for idx, (state, action) in enumerate(candidates):
            score = get_expected_fisher_information_gain(
                state, action, policy, ensemble_models, 
                state_scaler, action_scaler, device, gamma
            )
            efig_scores.append(score)
            
            if verbose and (idx + 1) % 500 == 0:
                print(f"     Processed {idx + 1}/{len(candidates)} candidates")
                
        efig_scores = np.array(efig_scores)
        top_indices = np.argsort(efig_scores)[-n_queries:][::-1]
        selected = [candidates[i] for i in top_indices]
        
        if verbose:
            print(f"   ✓ Selected {len(selected)} queries")
            print(f"   EFIG range: [{efig_scores.min():.6f}, {efig_scores.max():.6f}]")
            print(f"   Selected mean EFIG: {efig_scores[top_indices].mean():.6f}")
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
    """
    from config import HF_PARAMS
    
    if params is None:
        params = HF_PARAMS
    
    if verbose:
        print(f"\n🔬 Querying HF model...")
        print(f"   Number of queries: {len(queries)}")
    
    estimated_time_per_query = 0.01
    estimated_total_time = len(queries) * estimated_time_per_query
    
    if verbose:
        print(f"   Estimated time: {format_time(estimated_total_time)}")
    
    if not budget_tracker.check_budget(estimated_total_time):
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
        next_state, elapsed_time = query_single_step(state, action, params)
        total_time += elapsed_time
        reward = calculate_reward(state, action, next_state)
        
        new_transitions.append({
            'state': state,
            'action': action,
            'next_state': next_state,
            'reward': reward
        })
        
        if verbose and (idx + 1) % 100 == 0:
            print(f"     Queried {idx + 1}/{len(queries)} | Time: {format_time(total_time)}")
    
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
    """Augment dataset with new transitions."""
    if verbose:
        print(f"\n📊 Augmenting dataset...")
        print(f"   Old size: {len(old_dataset)}")
        print(f"   New transitions: {len(new_transitions)}")
    
    new_df = pd.DataFrame(new_transitions)
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
        description='Active learning: select and query optimal state-action pairs using EFIG'
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
    parser.add_argument('--selection_method', type=str, default='fisher_information',
                       choices=['fisher_information', 'uncertainty', 'random'],
                       help='Query selection method')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'])
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ACTIVE LEARNING (EXPECTED FISHER INFORMATION GAIN)")
    print("=" * 70)
    
    n_queries = args.n_queries or AL_CONFIG['n_queries_per_iteration']
    n_ensemble = AL_CONFIG['n_ensemble_models']
    
    budget_tracker = BudgetTracker(
        budget_file=Path('budget.json'),
        max_budget_seconds=28800  # 8 hours
    )
    
    print(f"\n📂 Loading dataset from: {args.dataset_path}")
    old_dataset = load_dataset(Path(args.dataset_path))
    
    state_scaler = load_scaler(Path(args.state_scaler_path))
    action_scaler = load_scaler(Path(args.action_scaler_path))
    
    print(f"\n🧠 Loading PPO Policy...")
    policy = PPO.load(args.policy_path, device=args.device)
    
    print(f"\n🏗️  Loading ensemble models...")
    ensemble_models = []
    ensemble_dir = Path(args.ensemble_dir)
    
    for i in range(n_ensemble):
        model_path = get_surrogate_path(args.version, ensemble_index=i)
        if not model_path.exists():
            continue
        
        model = SurrogateModel(state_dim=STATE_DIM, action_dim=ACTION_DIM)
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        model.to(args.device)
        model.eval()
        ensemble_models.append(model)
        print(f"   ✓ Loaded: {model_path.name}")
    
    # Step 1: Generate candidates
    candidates = generate_candidate_queries(
        policy=policy,
        surrogate_path=get_surrogate_path(args.version, ensemble_index=0),
        state_scaler_path=Path(args.state_scaler_path),
        action_scaler_path=Path(args.action_scaler_path),
    )
    
    # Step 2: Select best queries
    selected_queries = select_best_queries(
        candidates=candidates,
        ensemble_models=ensemble_models,
        policy=policy,
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
    
    output_path = Path(args.output_path)
    save_dataset(augmented_dataset, output_path)
    
    print("\n" + "=" * 70)
    print("✅ ACTIVE LEARNING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
