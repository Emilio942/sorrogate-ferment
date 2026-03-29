
import torch
import numpy as np
from pathlib import Path
from src.surrogate_env import SurrogateEnv
from src.config import get_surrogate_path, get_scaler_path

def test_improvements():
    print("Testing mathematical improvements...")
    
    # Paths for version 0 (assuming they might not exist, we just check logic)
    # If they don't exist, this test will fail gracefully with FileNotFoundError
    surr_path = get_surrogate_path(0).parent
    state_scaler = get_scaler_path('state', 0)
    action_scaler = get_scaler_path('action', 0)
    
    try:
        env = SurrogateEnv(
            surrogate_model_path=surr_path,
            state_scaler_path=state_scaler,
            action_scaler_path=action_scaler,
            use_ensemble=True
        )
        
        obs, _ = env.reset()
        print(f"Initial state: {obs}")
        
        # Test a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: Reward={reward:.4f}, Uncertainty={info['uncertainty']:.6f}, S={obs[1]:.2f}")
            
    except FileNotFoundError:
        print("Models not found. Please run iteration 0 of the experiment first.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_improvements()
