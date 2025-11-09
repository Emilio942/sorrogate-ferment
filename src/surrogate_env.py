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
    """Gymnasium environment using surrogate model for dynamics.
    
    The environment:
    - Uses a trained surrogate model to predict state transitions
    - Applies proper scaling/unscaling of states and actions
    - Calculates rewards based on the multi-objective function
    - Supports episodic interaction with configurable horizon
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        surrogate_model_path: Path,
        state_scaler_path: Path,
        action_scaler_path: Path,
        initial_states: np.ndarray = None,
        horizon: int = EPISODE_HORIZON,
        device: str = 'cpu'
    ):
        """Initialize surrogate environment.
        
        Args:
            surrogate_model_path: Path to trained surrogate model
            state_scaler_path: Path to fitted state scaler
            action_scaler_path: Path to fitted action scaler
            initial_states: Array of initial states to sample from (optional)
            horizon: Episode horizon (max steps)
            device: Device for model inference
        """
        super().__init__()
        
        self.horizon = horizon
        self.device = device
        self.current_step = 0
        
        # Load scalers
        print(f"Loading state scaler from: {state_scaler_path}")
        self.state_scaler = load_scaler(state_scaler_path)
        
        print(f"Loading action scaler from: {action_scaler_path}")
        self.action_scaler = load_scaler(action_scaler_path)
        
        # Load surrogate model
        print(f"Loading surrogate model from: {surrogate_model_path}")
        self.surrogate_model = SurrogateModel(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM
        )
        self.surrogate_model.load_state_dict(
            torch.load(surrogate_model_path, map_location=device)
        )
        self.surrogate_model.to(device)
        self.surrogate_model.eval()  # Set to evaluation mode
        
        # Initial states
        if initial_states is not None:
            self.initial_states = initial_states
        else:
            # Load default initial states from config ranges
            from config import INITIAL_STATE_RANGES
            from hf_model import sample_initial_state
            # Sample a pool of initial states
            self.initial_states = np.array([
                sample_initial_state(INITIAL_STATE_RANGES) 
                for _ in range(100)
            ])
        
        print(f"Initial states pool: {len(self.initial_states)} states")
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=ACTION_SPACE_LOW,
            high=ACTION_SPACE_HIGH,
            shape=(ACTION_DIM,),
            dtype=np.float32
        )
        
        # State space: [biomass, substrate] - use generous bounds
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([50.0, 100.0]),  # Generous upper bounds
            shape=(STATE_DIM,),
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        
        print(f"Surrogate environment initialized")
        print(f"  Action space: {self.action_space}")
        print(f"  Observation space: {self.observation_space}")
        print(f"  Horizon: {self.horizon}")
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state.
        
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Sample random initial state
        idx = np.random.randint(len(self.initial_states))
        self.state = self.initial_states[idx].copy().astype(np.float32)
        self.current_step = 0
        
        return self.state, {}
    
    def step(self, action: np.ndarray):
        """Execute one step in the environment.
        
        Args:
            action: Action to take [substrate_addition]
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Ensure action is numpy array and correct shape
        action = np.array(action, dtype=np.float32).reshape(-1)
        
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Store current state for reward calculation
        current_state = self.state.copy()
        
        # 1. Scale state and action
        state_scaled = self.state_scaler.transform(self.state.reshape(1, -1)).flatten()
        action_scaled = self.action_scaler.transform(action.reshape(1, -1)).flatten()
        
        # 2. Concatenate input: [state, action]
        input_tensor = torch.FloatTensor(
            np.concatenate([state_scaled, action_scaled])
        ).unsqueeze(0).to(self.device)
        
        # 3. Predict next state (scaled)
        with torch.no_grad():
            predicted_next_state_scaled = self.surrogate_model(input_tensor)
            predicted_next_state_scaled = predicted_next_state_scaled.cpu().numpy()
        
        # 4. Unscale predicted next state
        next_state = self.state_scaler.inverse_transform(
            predicted_next_state_scaled
        ).flatten().astype(np.float32)
        
        # 5. Ensure physical constraints (non-negative values)
        next_state = np.clip(next_state, 0.0, None)
        
        # 6. Calculate reward
        reward = calculate_reward(current_state, action, next_state, REWARD_WEIGHTS)
        
        # 7. Update state
        self.state = next_state
        self.current_step += 1
        
        # 8. Check termination conditions
        terminated = False
        truncated = self.current_step >= self.horizon
        
        # Early termination if biomass or substrate depleted
        if self.state[0] < 1e-6 or self.state[1] < 0:
            terminated = True
        
        info = {
            'step': self.current_step,
            'biomass': self.state[0],
            'substrate': self.state[1]
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
