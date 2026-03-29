"""
Koopman Model Predictive Control (MPC) and LQR baseline.
Replaces the PPO agent with a globally optimal controller based on the Deep Koopman Autoencoder.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from scipy.linalg import solve_discrete_are

# We use cvxpy for the Explicit MPC Quadratic Programming
try:
    import cvxpy as cp
except ImportError:
    print("cvxpy not installed. Please run: pip install cvxpy")
    sys.exit(1)

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    STATE_DIM, ACTION_DIM, HF_PARAMS, ACTION_BOUNDS,
    get_surrogate_path, get_scaler_path
)
from utils import load_dataset, load_scaler
from surrogate_koopman import DeepKoopmanAutoencoder
from data_builder import calculate_reward

# ============================================================================
# 1. REWARD REGRESSION (Quadratic Approximation)
# ============================================================================

def fit_reward_matrices(
    model: DeepKoopmanAutoencoder, 
    dataset: pd.DataFrame, 
    state_scaler, 
    action_scaler,
    device: str = 'cpu'
):
    """
    Fits the Q and R matrices in the latent space to approximate the true,
    non-quadratic reward.
    
    r(x, u) \approx - (z^T Q z + u^T R u)
    """
    print("\n📊 Fitting Quadratic Reward Surrogate (Q, R)...")
    model.eval()
    
    # We only need a subset to fit the matrices
    df_sample = dataset.sample(min(len(dataset), 5000))
    
    # 1. Get raw states, actions, and compute true rewards
    raw_states = np.array(df_sample['state'].tolist())
    raw_actions = np.array(df_sample['action'].tolist())
    raw_next_states = np.array(df_sample['next_state'].tolist())
    
    true_rewards = []
    for s, a, ns in zip(raw_states, raw_actions, raw_next_states):
        true_rewards.append(calculate_reward(s, a, ns))
    true_rewards = np.array(true_rewards)
    
    # We fit: -R_true(x, u) = z^T Q z + u^T R u
    target_neg_r = -true_rewards
    
    # 2. Get latent states z
    states_scaled = state_scaler.transform(raw_states).astype(np.float32)
    actions_scaled = action_scaler.transform(raw_actions).astype(np.float32)
    
    with torch.no_grad():
        z_t = model.encoder(torch.FloatTensor(states_scaled).to(device)).cpu().numpy()
        
    N_samples = len(z_t)
    latent_dim = model.latent_dim
    action_dim = ACTION_DIM
    
    # 3. Form the Design Matrix Phi
    # We need to vectorize z^T Q z and u^T R u.
    # z^T Q z = tr(z z^T Q) = vec(z z^T)^T vec(Q)
    
    # Number of unique elements in symmetric matrices
    # But for simplicity, we solve for diagonal Q and R first, which is often sufficient 
    # and strictly enforces PSD constraint trivially by taking abs() later if needed,
    # or solving bounded least squares.
    
    # Let's solve for diagonal Q and R to ensure Q>=0, R>0 easily without full SDP.
    Phi_Q = z_t ** 2           # [N_samples, latent_dim]
    Phi_R = actions_scaled ** 2 # [N_samples, action_dim]
    
    Phi = np.hstack([Phi_Q, Phi_R]) # [N_samples, latent_dim + action_dim]
    
    # Solve non-negative least squares to ensure Q, R >= 0
    from scipy.optimize import nnls
    theta, residual = nnls(Phi, target_neg_r)
    
    # Extract diagonal elements
    diag_Q = theta[:latent_dim]
    diag_R = theta[latent_dim:]
    
    # To ensure R is strictly positive definite
    diag_R = np.clip(diag_R, 1e-4, None)
    
    Q = np.diag(diag_Q)
    R = np.diag(diag_R)
    
    print(f"   ✓ Q matrix shape: {Q.shape}, Trace: {np.trace(Q):.4f}")
    print(f"   ✓ R matrix shape: {R.shape}, Trace: {np.trace(R):.4f}")
    print(f"   ✓ Residual error: {residual:.4f}")
    
    return Q, R


# ============================================================================
# 2. EXPLICIT MPC (Quadratic Programming)
# ============================================================================

class KoopmanMPC:
    """
    Model Predictive Control operating on the linear latent Koopman space.
    Solves a constrained QP at each step.
    """
    def __init__(
        self, 
        model: DeepKoopmanAutoencoder, 
        Q: np.ndarray, 
        R: np.ndarray,
        action_scaler,
        horizon: int = 5,
        device: str = 'cpu'
    ):
        self.model = model
        self.action_scaler = action_scaler
        self.device = device
        self.N = horizon
        
        # Get A and B from the model
        self.A = model.A.detach().cpu().numpy()
        self.B = model.B.detach().cpu().numpy()
        self.Q = Q
        self.R = R
        
        # Action limits in SCALED space
        # Original limits
        u_min_raw = np.array([ACTION_BOUNDS['feed_rate'][0]])
        u_max_raw = np.array([ACTION_BOUNDS['feed_rate'][1]])
        
        self.u_min = action_scaler.transform(u_min_raw.reshape(1, -1)).flatten()[0]
        self.u_max = action_scaler.transform(u_max_raw.reshape(1, -1)).flatten()[0]
        
        # Build Lifted Matrices
        self._build_lifted_matrices()
        
    def _build_lifted_matrices(self):
        """Builds H and F matrices for the QP."""
        A = self.A
        B = self.B
        N = self.N
        n_z = A.shape[0]
        n_u = B.shape[1]
        
        # Build block matrices Az and Bu
        # z_traj = A_l * z_0 + B_l * U
        A_l = np.zeros((N * n_z, n_z))
        B_l = np.zeros((N * n_z, N * n_u))
        
        for i in range(N):
            A_l[i*n_z:(i+1)*n_z, :] = np.linalg.matrix_power(A, i+1)
            for j in range(i+1):
                B_l[i*n_z:(i+1)*n_z, j*n_u:(j+1)*n_u] = np.linalg.matrix_power(A, i-j) @ B
                
        # Build block Q and R
        Q_l = np.kron(np.eye(N), self.Q)
        R_l = np.kron(np.eye(N), self.R)
        
        # Cost: U^T (B_l^T Q_l B_l + R_l) U + 2 * z_0^T A_l^T Q_l B_l U
        self.H = B_l.T @ Q_l @ B_l + R_l
        self.F = A_l.T @ Q_l @ B_l
        
    def get_action(self, z_curr: np.ndarray) -> np.ndarray:
        """Solves the QP to get the optimal next action in scaled space."""
        # Define CPX variables
        U = cp.Variable(self.N * ACTION_DIM)
        
        # Define objective: 1/2 U^T H U + (z_0^T F) U
        # Note: cvxpy quad_form is x^T P x, we need to ensure H is strictly PSD
        # Adding small epsilon to diagonal to ensure numerical PSD for CVXPY
        H_psd = self.H + np.eye(self.H.shape[0]) * 1e-6
        
        # linear term: f_T * U
        f_T = z_curr.T @ self.F
        
        objective = cp.Minimize(cp.quad_form(U, H_psd) + f_T @ U)
        
        # Define constraints
        constraints = [
            U >= self.u_min,
            U <= self.u_max
        ]
        
        # Solve QP
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.OSQP, warm_start=True)
            if U.value is None:
                # Fallback to zero action if solver fails
                return np.array([self.u_min])
            
            # Extract first action
            u_opt_scaled = U.value[:ACTION_DIM]
            return u_opt_scaled
        except Exception as e:
            print(f"QP Solver failed: {e}")
            return np.array([self.u_min])


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train LQR/MPC controller using Deep Koopman Autoencoder'
    )
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset for fitting Q and R')
    parser.add_argument('--koopman_path', type=str, required=True,
                       help='Path to trained Koopman Autoencoder')
    parser.add_argument('--state_scaler_path', type=str, required=True,
                       help='Path to state scaler')
    parser.add_argument('--action_scaler_path', type=str, required=True,
                       help='Path to action scaler')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--horizon', type=int, default=5,
                       help='MPC lookahead horizon')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("KOOPMAN LQR & MPC INITIALIZATION")
    print("=" * 70)
    
    # 1. Load Data & Scalers
    print("\n📂 Loading scalers and data...")
    dataset = load_dataset(Path(args.dataset_path))
    state_scaler = load_scaler(Path(args.state_scaler_path))
    action_scaler = load_scaler(Path(args.action_scaler_path))
    
    # 2. Load Koopman Model
    print(f"\n🧠 Loading Deep Koopman Autoencoder...")
    # Assuming we know the architecture params from training
    model = DeepKoopmanAutoencoder(latent_dim=32, hidden_dim=128)
    model.load_state_dict(torch.load(args.koopman_path, map_location=args.device))
    model.to(args.device)
    model.eval()
    
    # 3. Fit Q and R Matrices
    Q, R = fit_reward_matrices(model, dataset, state_scaler, action_scaler, device=args.device)
    
    # 4. Initialize MPC
    print(f"\n⚙️  Initializing Explicit MPC (Horizon N={args.horizon})...")
    mpc = KoopmanMPC(model, Q, R, action_scaler, horizon=args.horizon, device=args.device)
    
    print("\n✓ MPC Ready! You can now use `mpc.get_action(z_curr)` in the environment loop.")
    print("=" * 70)

if __name__ == '__main__':
    main()