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
    horizon: int = EPISODE_HORIZON,
    use_ensemble: bool = True
):
    """Run validation comparison between HF and Surrogate (Ensemble)."""
    print(f"Loading surrogate model from {surrogate_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model(s)
    models = []
    surrogate_path = Path(surrogate_path)
    if use_ensemble:
        if surrogate_path.is_file():
            model_dir = surrogate_path.parent
            model_pattern = surrogate_path.name.replace("_best.pth", "_ens_*.pth")
            model_paths = list(model_dir.glob(model_pattern))
            if not model_paths:
                model_paths = [surrogate_path]
        else:
            model_paths = list(surrogate_path.glob("surrogate_v*_ens_*.pth"))
            if not model_paths:
                model_paths = list(surrogate_path.glob("*.pth"))
        
        print(f"Loading ensemble of {len(model_paths)} models")
        for p in model_paths:
            m = SurrogateModel(STATE_DIM, ACTION_DIM)
            m.load_state_dict(torch.load(p, map_location=device))
            m.to(device)
            m.eval()
            models.append(m)
    else:
        m = SurrogateModel(STATE_DIM, ACTION_DIM)
        m.load_state_dict(torch.load(surrogate_path, map_location=device))
        m.to(device)
        m.eval()
        models.append(m)
    
    # Load scalers
    state_scaler = load_scaler(state_scaler_path)
    action_scaler = load_scaler(action_scaler_path)
    
    # 1. Define a test scenario (Constant Feed)
    print("\nRunning Test Scenario: Constant Feed")
    initial_state = np.array([0.1, 20.0, 1.0])
    constant_action = np.array([1.0])
    
    hf_states = [initial_state]
    surr_means = [initial_state]
    surr_stds = [np.zeros(STATE_DIM)]
    
    current_hf_state = initial_state.copy()
    current_surr_mean = initial_state.copy()
    
    dt = HF_PARAMS['DT']
    
    for t in range(horizon):
        action = constant_action
        
        # --- HF Step ---
        dydt = hf_dynamics(0, current_hf_state, action, HF_PARAMS)
        next_hf_state = current_hf_state + dydt * dt
        next_hf_state = np.clip(next_hf_state, 0.0, None)
        hf_states.append(next_hf_state)
        current_hf_state = next_hf_state
        
        # --- Surrogate Ensemble Step ---
        s_scaled = state_scaler.transform(current_surr_mean.reshape(1, -1)).flatten()
        a_scaled = action_scaler.transform(action.reshape(1, -1)).flatten()
        inp = torch.FloatTensor(np.concatenate([s_scaled, a_scaled])).unsqueeze(0).to(device)
        
        all_preds_scaled = []
        with torch.no_grad():
            for m in models:
                p_scaled = m(inp).cpu().numpy()
                all_preds_scaled.append(p_scaled.flatten())
        
        all_preds_scaled = np.array(all_preds_scaled)
        mean_scaled = np.mean(all_preds_scaled, axis=0)
        std_scaled = np.std(all_preds_scaled, axis=0)
        
        # Unscale mean and approximate unscaled std
        next_surr_mean = state_scaler.inverse_transform(mean_scaled.reshape(1, -1)).flatten()
        # Scale the std by the state_scaler's scale factor (approximation)
        if hasattr(state_scaler, 'scale_'):
            next_surr_std = std_scaled * state_scaler.scale_
        else:
            next_surr_std = std_scaled
            
        next_surr_mean = np.clip(next_surr_mean, 0.0, None)
        surr_means.append(next_surr_mean)
        surr_stds.append(next_surr_std)
        current_surr_mean = next_surr_mean

    hf_states = np.array(hf_states)
    surr_means = np.array(surr_means)
    surr_stds = np.array(surr_stds)
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ['Biomass [g/L]', 'Substrate [g/L]', 'Volume [L]']
    
    for i in range(3):
        axes[i].plot(hf_states[:, i], 'k-', label='HF Model', linewidth=2)
        axes[i].plot(surr_means[:, i], 'r--', label='Surrogate Mean', linewidth=2)
        # Uncertainty band (2 sigma)
        axes[i].fill_between(
            range(len(surr_means)),
            surr_means[:, i] - 2 * surr_stds[:, i],
            surr_means[:, i] + 2 * surr_stds[:, i],
            color='r', alpha=0.2, label='Uncertainty (2σ)'
        )
        axes[i].set_title(titles[i])
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plot_path = output_dir / "ensemble_validation.png"
    plt.savefig(plot_path)
    print(f"Ensemble validation plot saved to {plot_path}")
    
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
