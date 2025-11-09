"""
High-Fidelity fermentation model based on Monod kinetics.
Implements ODE dynamics and episode simulation.
"""
import time
from typing import Callable, Dict, Tuple, List
import numpy as np
from scipy.integrate import solve_ivp
from config import HF_PARAMS, EPISODE_HORIZON, INITIAL_STATE_RANGES, MAX_SUBSTRATE_ADDITION


# ============================================================================
# ODE DYNAMICS (Monod Kinetics)
# ============================================================================

def hf_dynamics(t: float, y: np.ndarray, action: np.ndarray, 
                params: Dict[str, float]) -> np.ndarray:
    """High-fidelity fermentation dynamics based on Monod kinetics.
    
    Implements the differential equations for:
    - Biomass growth (Monod kinetics with death rate)
    - Substrate consumption
    
    Args:
        t: Time (not used directly but required by solve_ivp)
        y: State vector [biomass, substrate]
        action: Control input [substrate_addition_rate]
        params: Dictionary with parameters (MU_MAX, K_S, YXS, K_D)
        
    Returns:
        Time derivatives [dX/dt, dS/dt]
    """
    # Extract state variables
    X = max(y[0], 1e-10)  # Biomass [g/L], ensure positive
    S = max(y[1], 0.0)     # Substrate [g/L], ensure non-negative
    
    # Extract parameters
    mu_max = params['MU_MAX']  # Maximum specific growth rate [1/h]
    K_s = params['K_S']        # Substrate saturation constant [g/L]
    Y_xs = params['YXS']       # Yield coefficient [g biomass/g substrate]
    k_d = params['K_D']        # Death rate constant [1/h]
    
    # Extract control action
    substrate_addition = max(action[0], 0.0)  # Substrate addition rate [g/L/h]
    
    # Monod kinetics: specific growth rate
    mu = mu_max * S / (K_s + S)
    
    # Differential equations
    dX_dt = mu * X - k_d * X                    # Biomass: growth - death
    dS_dt = -mu * X / Y_xs + substrate_addition  # Substrate: consumption + addition
    
    return np.array([dX_dt, dS_dt])


# ============================================================================
# EPISODE SIMULATION
# ============================================================================

def simulate_episode(
    initial_state: np.ndarray,
    action_policy_func: Callable[[np.ndarray], np.ndarray],
    horizon: int = EPISODE_HORIZON,
    params: Dict[str, float] = None,
    dt: float = None
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]], float]:
    """Simulate one episode using the high-fidelity model.
    
    NOTE: This function does NOT check budget. The caller is responsible for
    budget checking before calling this function.
    
    Args:
        initial_state: Initial state [biomass, substrate]
        action_policy_func: Function that takes state and returns action
        horizon: Number of time steps
        params: Model parameters (default: HF_PARAMS from config)
        dt: Time step size (default: from params)
        
    Returns:
        Tuple of (trajectory, elapsed_time) where:
            - trajectory: List of (state, action, next_state, reward) tuples
            - elapsed_time: Wall-clock time in seconds
    """
    if params is None:
        params = HF_PARAMS
    if dt is None:
        dt = params['DT']
    
    trajectory = []
    current_state = np.array(initial_state, dtype=float)
    
    # Start timing
    start_time = time.time()
    
    for step in range(horizon):
        # Get action from policy
        action = action_policy_func(current_state)
        action = np.clip(action, 0.0, MAX_SUBSTRATE_ADDITION)
        
        # Integrate ODE for one time step
        def dynamics_with_action(t, y):
            return hf_dynamics(t, y, action, params)
        
        # Solve ODE from t to t+dt
        sol = solve_ivp(
            dynamics_with_action,
            t_span=[0, dt],
            y0=current_state,
            method='RK45',
            dense_output=False,
            max_step=dt/10
        )
        
        # Get next state (final point of integration)
        next_state = sol.y[:, -1]
        
        # Ensure state bounds (non-negative values)
        next_state = np.clip(next_state, 0.0, None)
        
        # Calculate reward (will be implemented in reward function)
        reward = 0.0  # Placeholder, will be computed by environment/evaluation
        
        # Store transition
        trajectory.append((
            current_state.copy(),
            action.copy(),
            next_state.copy(),
            reward
        ))
        
        # Update state
        current_state = next_state
        
        # Early termination if substrate or biomass depleted
        if next_state[0] < 1e-6 or next_state[1] < 0:
            break
    
    # Calculate elapsed wall-clock time
    elapsed_time = time.time() - start_time
    
    return trajectory, elapsed_time


# ============================================================================
# INITIAL STATE SAMPLING
# ============================================================================

def sample_initial_state(ranges: Dict[str, Tuple[float, float]] = None) -> np.ndarray:
    """Sample a random initial state within specified ranges.
    
    Args:
        ranges: Dictionary with ranges for each state variable
                (default: INITIAL_STATE_RANGES from config)
        
    Returns:
        Initial state vector [biomass, substrate]
    """
    if ranges is None:
        ranges = INITIAL_STATE_RANGES
    
    biomass = np.random.uniform(*ranges['biomass'])
    substrate = np.random.uniform(*ranges['substrate'])
    
    return np.array([biomass, substrate])


# ============================================================================
# SINGLE-STEP QUERY (for Active Learning)
# ============================================================================

def query_single_step(
    state: np.ndarray,
    action: np.ndarray,
    params: Dict[str, float] = None,
    dt: float = None
) -> Tuple[np.ndarray, float]:
    """Query the HF model for a single state transition.
    
    Used in active learning to get ground truth for specific (state, action) pairs.
    
    Args:
        state: Current state [biomass, substrate]
        action: Control action [substrate_addition]
        params: Model parameters (default: HF_PARAMS from config)
        dt: Time step size (default: from params)
        
    Returns:
        Tuple of (next_state, elapsed_time)
    """
    if params is None:
        params = HF_PARAMS
    if dt is None:
        dt = params['DT']
    
    start_time = time.time()
    
    # Integrate ODE for one time step
    def dynamics_with_action(t, y):
        return hf_dynamics(t, y, action, params)
    
    sol = solve_ivp(
        dynamics_with_action,
        t_span=[0, dt],
        y0=state,
        method='RK45',
        dense_output=False,
        max_step=dt/10
    )
    
    next_state = sol.y[:, -1]
    next_state = np.clip(next_state, 0.0, None)
    
    elapsed_time = time.time() - start_time
    
    return next_state, elapsed_time


# ============================================================================
# TESTING AND BENCHMARKING
# ============================================================================

def benchmark_episode_time(n_episodes: int = 10, horizon: int = EPISODE_HORIZON) -> Dict:
    """Benchmark the time required for HF episodes.
    
    Args:
        n_episodes: Number of episodes to run for benchmarking
        horizon: Number of steps per episode
        
    Returns:
        Dictionary with timing statistics
    """
    print(f"Benchmarking HF model with {n_episodes} episodes...")
    
    def random_policy(state):
        return np.array([np.random.uniform(0, MAX_SUBSTRATE_ADDITION)])
    
    times = []
    for i in range(n_episodes):
        initial_state = sample_initial_state()
        _, elapsed_time = simulate_episode(
            initial_state,
            random_policy,
            horizon=horizon
        )
        times.append(elapsed_time)
        print(f"  Episode {i+1}/{n_episodes}: {elapsed_time:.3f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    total_time = np.sum(times)
    
    # Estimate how many episodes fit in 8h budget
    budget_hours = 8
    budget_seconds = budget_hours * 3600
    estimated_max_episodes = int(budget_seconds / avg_time)
    
    stats = {
        'n_episodes': n_episodes,
        'avg_time_per_episode': avg_time,
        'std_time_per_episode': std_time,
        'total_time': total_time,
        'budget_hours': budget_hours,
        'budget_seconds': budget_seconds,
        'estimated_max_episodes_in_budget': estimated_max_episodes
    }
    
    print(f"\nBenchmark Results:")
    print(f"  Average time per episode: {avg_time:.3f} ± {std_time:.3f}s")
    print(f"  Estimated max episodes in {budget_hours}h budget: {estimated_max_episodes}")
    
    return stats
