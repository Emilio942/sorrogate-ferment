"""
Surrogate environment: Gymnasium wrapper for surrogate model.
Allows RL training on the learned dynamics model.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    STATE_DIM, ACTION_DIM, EPISODE_HORIZON,
    ACTION_SPACE_LOW, ACTION_SPACE_HIGH,
    REWARD_WEIGHTS
)
from utils import load_scaler
from surrogate_model import SurrogateModel
from data_builder import calculate_reward


# ============================================================================
# SURROGATE ENVIRONMENT
# ============================================================================

class SurrogateEnv(gym.Env):
    """Gymnasium environment using an ensemble of surrogate models for dynamics.
    
    The environment:
    - Uses an ensemble of trained surrogate models to predict state transitions
    - Computes the mean prediction for transitions to reduce model bias
    - Quantifies uncertainty as the variance across ensemble members
    - Applies proper scaling/unscaling of states and actions
    - Calculates rewards based on the multi-objective function
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        surrogate_model_path: Path,
        state_scaler_path: Path,
        action_scaler_path: Path,
        initial_states: np.ndarray = None,
        horizon: int = EPISODE_HORIZON,
        device: str = 'cpu',
        use_ensemble: bool = True
    ):
        """Initialize surrogate environment.
        
        Args:
            surrogate_model_path: Path to surrogate model or directory containing ensemble
            state_scaler_path: Path to fitted state scaler
            action_scaler_path: Path to fitted action scaler
            initial_states: Array of initial states to sample from (optional)
            horizon: Episode horizon (max steps)
            device: Device for model inference
            use_ensemble: If True, load all ensemble members in the same directory
        """
        super().__init__()
        
        self.horizon = horizon
        self.device = device
        self.current_step = 0
        
        # Load scalers
        self.state_scaler = load_scaler(state_scaler_path)
        self.action_scaler = load_scaler(action_scaler_path)
        
        # Load surrogate model(s)
        self.models = []
        surrogate_model_path = Path(surrogate_model_path)
        
        if use_ensemble:
            # Check if path is a file or directory
            if surrogate_model_path.is_file():
                model_dir = surrogate_model_path.parent
                model_pattern = surrogate_model_path.name.replace("_best.pth", "_ens_*.pth")
                model_paths = list(model_dir.glob(model_pattern))
                if not model_paths:
                    model_paths = [surrogate_model_path]
            else:
                model_paths = list(surrogate_model_path.glob("surrogate_v*_ens_*.pth"))
                if not model_paths:
                    model_paths = list(surrogate_model_path.glob("*.pth"))
            
            print(f"Loading ensemble of {len(model_paths)} models from {surrogate_model_path.parent}")
            for p in model_paths:
                model = SurrogateModel(state_dim=STATE_DIM, action_dim=ACTION_DIM)
                model.load_state_dict(torch.load(p, map_location=device))
                model.to(device)
                model.eval()
                self.models.append(model)
        else:
            model = SurrogateModel(state_dim=STATE_DIM, action_dim=ACTION_DIM)
            model.load_state_dict(torch.load(surrogate_model_path, map_location=device))
            model.to(device)
            model.eval()
            self.models.append(model)
        
        # Initial states
        if initial_states is not None:
            self.initial_states = initial_states
        else:
            from config import INITIAL_STATE_RANGES
            from hf_model import sample_initial_state
            self.initial_states = np.array([
                sample_initial_state(INITIAL_STATE_RANGES) 
                for _ in range(100)
            ])
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=ACTION_SPACE_LOW,
            high=ACTION_SPACE_HIGH,
            shape=(ACTION_DIM,),
            dtype=np.float32
        )
        
        # State space: [biomass, substrate, volume]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([50.0, 100.0, 10.0]),
            shape=(STATE_DIM,),
            dtype=np.float32
        )
        
        self.state = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = np.random.randint(len(self.initial_states))
        self.state = self.initial_states[idx].copy().astype(np.float32)
        self.current_step = 0
        return self.state, {}
    
    def step(self, action: np.ndarray):
        from scipy.optimize import root
        from config import HF_PARAMS
        
        action = np.array(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        current_state = self.state.copy()
        
        # 1. Scale state and action
        state_scaled = self.state_scaler.transform(self.state.reshape(1, -1)).flatten()
        action_scaled = self.action_scaler.transform(action.reshape(1, -1)).flatten()
        
        dt = HF_PARAMS['DT']
        a_tensor = torch.FloatTensor(action_scaled).unsqueeze(0).to(self.device)
        
        # 2. Semi-Implicit Euler via Root Finding
        def residual(s_guess_np):
            # Equation: s_next - s_curr - dt * f(s_next, a) = 0
            s_tensor = torch.FloatTensor(s_guess_np).unsqueeze(0).to(self.device)
            input_tensor = torch.cat([s_tensor, a_tensor], dim=-1)
            
            with torch.no_grad():
                preds = [model(input_tensor) for model in self.models]
                mean_ds_dt = torch.mean(torch.stack(preds), dim=0).cpu().numpy().flatten()
            
            return s_guess_np - state_scaled - dt * mean_ds_dt

        # Use scipy.optimize.root with hybr method (Powell's method)
        # We start the guess at the current state
        sol = root(residual, state_scaled, method='hybr')
        s_next_scaled = sol.x
        
        # 3. Compute uncertainty (variance) at the new state
        s_next_tensor = torch.FloatTensor(s_next_scaled).unsqueeze(0).to(self.device)
        input_tensor = torch.cat([s_next_tensor, a_tensor], dim=-1)
        with torch.no_grad():
            preds = [model(input_tensor) for model in self.models]
            preds_stack = torch.stack(preds) # [n_models, 1, state_dim]
            uncertainty = torch.mean(torch.var(preds_stack, dim=0)).item()
        
        # 4. Unscale predicted next state
        next_state = self.state_scaler.inverse_transform(
            s_next_scaled.reshape(1, -1)
        ).flatten().astype(np.float32)
        
        # 6. Ensure physical constraints
        next_state = np.clip(next_state, 0.0, None)
        
        # 7. Calculate reward
        reward = calculate_reward(current_state, action, next_state, REWARD_WEIGHTS)
        
        # 8. Update state
        self.state = next_state
        self.current_step += 1
        
        # 9. Termination conditions
        terminated = False
        truncated = self.current_step >= self.horizon
        
        if self.state[0] < 1e-6 or self.state[1] < 0:
            terminated = True
        
        info = {
            'step': self.current_step,
            'biomass': self.state[0],
            'substrate': self.state[1],
            'volume': self.state[2],
            'uncertainty': float(uncertainty)
        }
        
        return self.state, float(reward), terminated, truncated, info
    
    def render(self):
        """Render environment (not implemented)."""
        pass
    
    def close(self):
        """Close environment."""
        pass


# ============================================================================
# TESTING UTILITIES
# ============================================================================

def test_surrogate_env(
    surrogate_model_path: Path,
    state_scaler_path: Path,
    action_scaler_path: Path,
    n_episodes: int = 3
):
    """Test surrogate environment with random actions.
    
    Args:
        surrogate_model_path: Path to surrogate model
        state_scaler_path: Path to state scaler
        action_scaler_path: Path to action scaler
        n_episodes: Number of test episodes
    """
    print("=" * 70)
    print("TESTING SURROGATE ENVIRONMENT")
    print("=" * 70)
    
    # Create environment
    env = SurrogateEnv(
        surrogate_model_path,
        state_scaler_path,
        action_scaler_path
    )
    
    # Test gymnasium check_env
    from gymnasium.utils.env_checker import check_env
    print("\n🔍 Running gymnasium environment checks...")
    try:
        check_env(env.unwrapped, skip_render_check=True)
        print("✓ Environment passes all checks")
    except Exception as e:
        print(f"⚠ Environment check failed: {e}")
    
    # Run test episodes
    print(f"\n🎮 Running {n_episodes} test episodes with random actions...")
    
    for episode in range(n_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        print(f"\n  Episode {episode + 1}:")
        print(f"    Initial state: {state}")
        
        while True:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        print(f"    Final state: {next_state}")
        print(f"    Episode reward: {episode_reward:.3f}")
        print(f"    Episode length: {episode_length}")
    
    env.close()
    
    print("\n" + "=" * 70)
    print("✅ ENVIRONMENT TEST COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test surrogate environment')
    parser.add_argument('--surrogate_path', type=str, required=True)
    parser.add_argument('--state_scaler_path', type=str, required=True)
    parser.add_argument('--action_scaler_path', type=str, required=True)
    parser.add_argument('--n_episodes', type=int, default=3)
    
    args = parser.parse_args()
    
    test_surrogate_env(
        Path(args.surrogate_path),
        Path(args.state_scaler_path),
        Path(args.action_scaler_path),
        n_episodes=args.n_episodes
    )
