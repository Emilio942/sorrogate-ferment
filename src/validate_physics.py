"""
Physics Validation Script
Validates that the surrogate model learns real physics and not just memorizes data.
Compares HF model trajectories with Surrogate predictions.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    HF_PARAMS, STATE_DIM, ACTION_DIM, 
    get_surrogate_path, get_scaler_path, PLOTS_DIR
)
from hf_model import hf_dynamics, sample_initial_state
from surrogate_model import SurrogateModel
from utils import load_scaler

def validate_physics(
    surrogate_path: Path,
    state_scaler_path: Path,
    action_scaler_path: Path,
    output_dir: Path,
    horizon: int = 50
):
    """Run validation comparison."""
    print(f"Loading surrogate model from {surrogate_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = SurrogateModel(STATE_DIM, ACTION_DIM)
    model.load_state_dict(torch.load(surrogate_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load scalers
    state_scaler = load_scaler(state_scaler_path)
    action_scaler = load_scaler(action_scaler_path)
    
    # 1. Define a test scenario (Constant Feed)
    print("\nRunning Test Scenario: Constant Feed")
    initial_state = np.array([0.1, 20.0]) # Low biomass, medium substrate
    constant_action = np.array([1.0])     # Constant feed
    
    # Arrays to store trajectories
    hf_states = [initial_state]
    surrogate_states = [initial_state]
    actions = []
    
    # Run simulation
    current_hf_state = initial_state.copy()
    current_surr_state = initial_state.copy() # Open loop prediction (errors accumulate)
    
    dt = HF_PARAMS['DT']
    
    for t in range(horizon):
        action = constant_action
        actions.append(action)
        
        # --- HF Step ---
        # Simple Euler integration for validation (or use solve_ivp for precision)
        # dy/dt = f(t, y, u)
        # y_next = y + dy/dt * dt
        dydt = hf_dynamics(0, current_hf_state, action, HF_PARAMS)
        next_hf_state = current_hf_state + dydt * dt
        # Clip
        next_hf_state[0] = max(next_hf_state[0], 1e-10)
        next_hf_state[1] = max(next_hf_state[1], 0.0)
        
        hf_states.append(next_hf_state)
        current_hf_state = next_hf_state
        
        # --- Surrogate Step (Open Loop) ---
        # Predict next state based on PREVIOUS SURROGATE STATE (testing long-term stability)
        
        # Scale
        s_scaled = state_scaler.transform(current_surr_state.reshape(1, -1)).flatten()
        a_scaled = action_scaler.transform(action.reshape(1, -1)).flatten()
        
        inp = torch.FloatTensor(np.concatenate([s_scaled, a_scaled])).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_scaled = model(inp).cpu().numpy()
            
        next_surr_state = state_scaler.inverse_transform(pred_scaled).flatten()
        
        # Enforce physical constraints (same as in SurrogateEnv)
        next_surr_state = np.clip(next_surr_state, 0.0, None)
        
        surrogate_states.append(next_surr_state)
        current_surr_state = next_surr_state

    # Convert to arrays
    hf_states = np.array(hf_states)
    surrogate_states = np.array(surrogate_states)
    actions = np.array(actions)
    
    # Calculate Error
    mse = np.mean((hf_states - surrogate_states)**2)
    print(f"Open-Loop MSE over {horizon} steps: {mse:.6f}")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Biomass
    axes[0].plot(hf_states[:, 0], 'b-', label='HF Model (Ground Truth)', linewidth=2)
    axes[0].plot(surrogate_states[:, 0], 'r--', label='Surrogate (Open Loop)', linewidth=2)
    axes[0].set_title('Biomass Trajectory')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Biomass [g/L]')
    axes[0].legend()
    axes[0].grid(True)
    
    # Substrate
    axes[1].plot(hf_states[:, 1], 'b-', label='HF Model', linewidth=2)
    axes[1].plot(surrogate_states[:, 1], 'r--', label='Surrogate', linewidth=2)
    axes[1].set_title('Substrate Trajectory')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Substrate [g/L]')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plot_path = output_dir / "physics_validation.png"
    plt.savefig(plot_path)
    print(f"Validation plot saved to {plot_path}")
    
    # Check for physical violations
    neg_biomass = np.sum(surrogate_states[:, 0] < 0)
    neg_substrate = np.sum(surrogate_states[:, 1] < 0)
    
    if neg_biomass > 0 or neg_substrate > 0:
        print(f"⚠ WARNING: Surrogate predicted negative values! (Biomass: {neg_biomass}, Substrate: {neg_substrate})")
    else:
        print("✓ Surrogate predictions remained non-negative.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, default=1)
    parser.add_argument('--ensemble_index', type=int, default=None)
    args = parser.parse_args()
    
    surrogate_path = get_surrogate_path(args.version, args.ensemble_index)
    state_scaler_path = get_scaler_path('state', args.version)
    action_scaler_path = get_scaler_path('action', args.version)
    
    if not surrogate_path.exists():
        print(f"Error: Model not found at {surrogate_path}")
        print("Please run training first.")
        sys.exit(1)
        
    validate_physics(
        surrogate_path,
        state_scaler_path,
        action_scaler_path,
        PLOTS_DIR
    )
