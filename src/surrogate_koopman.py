"""
Deep Koopman Autoencoder (DKA) implementation.
Lifts non-linear states to a higher-dimensional latent space where dynamics are affine linear.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import sys

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))
from config import STATE_DIM, ACTION_DIM

# ============================================================================
# ARCHITECTURE
# ============================================================================

class Encoder(nn.Module):
    """Maps raw state to latent observable z."""
    def __init__(self, state_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim) # Linear output
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Decoder(nn.Module):
    """Maps latent observable z back to raw state."""
    def __init__(self, latent_dim: int, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim) # Linear output
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class DeepKoopmanAutoencoder(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, 
                 latent_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.encoder = Encoder(state_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, state_dim, hidden_dim)
        
        # Learnable linear matrices for continuous/discrete time dynamics
        # z_{t+1} = A z_t + B u_t
        self.A = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(latent_dim, action_dim) * 0.01)
        
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard AE forward pass."""
        z = self.encoder(x)
        return self.decoder(z)
    
    def project_stable(self):
        """Enforces A-stability (spectral radius < 1) by projecting eigenvalues."""
        with torch.no_grad():
            # Move to CPU for eigendecomposition if needed, but torch.linalg.eig works on GPU
            L, V = torch.linalg.eig(self.A)
            abs_L = torch.abs(L)
            
            # If any eigenvalue magnitude > 0.99, scale it down
            max_eig = torch.max(abs_L)
            if max_eig > 0.99:
                scale = 0.99 / max_eig
                # A_new = V * (L * scale) * V^-1
                L_scaled = L * scale
                # Reconstruct A (keeping it real)
                A_new = torch.real(V @ torch.diag(L_scaled) @ torch.linalg.inv(V))
                self.A.copy_(A_new)

# ============================================================================
# TRAINING
# ============================================================================

def train_koopman(
    model: DeepKoopmanAutoencoder,
    dataloader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-4,
    lambda_koop: float = 1.0,
    lambda_lin: float = 0.1,
    device: str = 'cpu'
):
    """Trains the Deep Koopman Autoencoder."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n🚀 Starting Koopman Training...")
    print(f"   Latent Dim: {model.latent_dim}")
    print(f"   Epochs: {epochs}")
    print(f"   Lambdas - Koop: {lambda_koop}, Lin: {lambda_lin}")
    
    # We assume dataloader yields (x_t, u_t, x_{t+1}) 
    # For now, we adapt the existing dataset logic where inputs are [state, action] and targets are next_state
    
    for epoch in range(epochs):
        model.train()
        total_rec_loss = 0.0
        total_koop_loss = 0.0
        total_lin_loss = 0.0
        
        for batch_idx, (inputs, next_states) in enumerate(dataloader):
            inputs, next_states = inputs.to(device), next_states.to(device)
            
            x_t = inputs[:, :STATE_DIM]
            u_t = inputs[:, STATE_DIM:]
            x_next = next_states # Assuming targets are the actual next states, not derivatives for this model
            
            optimizer.zero_grad()
            
            # 1. Encode
            z_t = model.encoder(x_t)
            z_next_target = model.encoder(x_next)
            
            # 2. Reconstruction Loss
            x_hat = model.decoder(z_t)
            L_rec = torch.mean((x_hat - x_t) ** 2)
            
            # 3. Latent Dynamics Prediction (Koopman Loss)
            # z_{t+1} = A z_t + B u_t
            # Matrix mult: [B, latent] @ [latent, latent]^T + [B, action] @ [action, latent]^T
            z_next_pred = torch.nn.functional.linear(z_t, model.A, bias=None) + \
                          torch.nn.functional.linear(u_t, model.B, bias=None)
            
            L_koop = torch.mean((z_next_pred - z_next_target) ** 2)
            
            # 4. Linear Regularizer (Optional, enforces encoder consistency)
            # Penalizes if encoding the next state doesn't lie on the affine transition
            L_lin = torch.mean((model.encoder(x_next) - z_next_pred.detach()) ** 2)
            
            # 5. Controllability Regularizer
            # Kalman Controllability Matrix C = [B, AB, A^2B, ..., A^{n-1}B]
            n_latent = model.latent_dim
            A_mat = model.A
            B_mat = model.B
            
            C_list = [B_mat]
            curr_term = B_mat
            # For large n_latent, computing all n_latent terms might be expensive, 
            # we can truncate to a smaller horizon or compute full. Let's do a modest horizon.
            horizon = min(n_latent, 10) 
            for _ in range(1, horizon):
                curr_term = A_mat @ curr_term
                C_list.append(curr_term)
                
            C_mat = torch.cat(C_list, dim=1) # [n_latent, horizon * action_dim]
            
            # Smallest singular value of C
            # Using SVD
            try:
                sigma_min = torch.linalg.svdvals(C_mat).min()
                eps = 1e-6
                # Log barrier to prevent sigma_min from approaching 0
                R_C = -torch.log(sigma_min + eps)
            except RuntimeError:
                # Fallback if SVD fails to converge
                R_C = torch.tensor(0.0, device=device)
            
            lambda_C = 0.01
            
            # Total Loss
            loss = L_rec + lambda_koop * L_koop + lambda_lin * L_lin + lambda_C * R_C
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            # Enforce stability on A
            model.project_stable()
            
            total_rec_loss += L_rec.item()
            total_koop_loss += L_koop.item()
            total_lin_loss += L_lin.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs} | "
                  f"Rec: {total_rec_loss/len(dataloader):.4f} | "
                  f"Koop: {total_koop_loss/len(dataloader):.4f} | "
                  f"Lin: {total_lin_loss/len(dataloader):.4f}")

    print("✓ Koopman Training Complete!")
    return model
