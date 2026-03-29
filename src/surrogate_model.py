"""
Surrogate model implementation: PyTorch dataset, architecture, and training.
Implements a Physics-Informed Neural ODE for state transition prediction
with Dissipative Jacobian Regularization for A-stability.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    SURROGATE_CONFIG, STATE_DIM, ACTION_DIM,
    HF_PARAMS, get_surrogate_path, get_scaler_path
)
from utils import load_dataset, load_scaler, save_model, load_model
from logger import ExperimentLogger


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class FermentationDataset(Dataset):
    """PyTorch dataset for fermentation state transitions.
    
    Loads dataset and applies scaling transformations.
    Targets are converted to scaled derivatives: ds/dt
    """
    
    def __init__(self, dataset_path: Path = None, state_scaler_path: Path = None, 
                 action_scaler_path: Path = None, bootstrap_sample: bool = False,
                 dataframe: pd.DataFrame = None):
        """Initialize dataset.
        
        Args:
            dataset_path: Path to dataset pickle file (optional if dataframe provided)
            state_scaler_path: Path to fitted state scaler
            action_scaler_path: Path to fitted action scaler
            bootstrap_sample: If True, perform bootstrap sampling (for ensemble)
            dataframe: Pre-loaded DataFrame (optional)
        """
        # Load dataset
        if dataframe is not None:
            self.df = dataframe.copy()
        elif dataset_path is not None:
            self.df = load_dataset(dataset_path)
        else:
            raise ValueError("Either dataset_path or dataframe must be provided")
        
        # Bootstrap sampling if requested (for ensemble training)
        if bootstrap_sample:
            n_samples = len(self.df)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            self.df = self.df.iloc[indices].reset_index(drop=True)
            print(f"   Bootstrap sampling: {n_samples} samples")
        
        # Load scalers
        self.state_scaler = load_scaler(state_scaler_path)
        self.action_scaler = load_scaler(action_scaler_path)
        
        # Extract and PRE-SCALE data for efficiency
        raw_states = np.array([s for s in self.df['state']])
        raw_actions = np.array([a for a in self.df['action']])
        raw_next_states = np.array([s for s in self.df['next_state']])
        
        print(f"   Scaling data...")
        self.states_scaled = self.state_scaler.transform(raw_states).astype(np.float32)
        self.actions_scaled = self.action_scaler.transform(raw_actions).astype(np.float32)
        self.next_states_scaled = self.state_scaler.transform(raw_next_states).astype(np.float32)
        
        # Concatenate inputs once
        self.inputs = np.concatenate([self.states_scaled, self.actions_scaled], axis=1)
        
        # Calculate scaled derivative ds/dt as the target
        dt = HF_PARAMS['DT']
        self.targets = (self.next_states_scaled - self.states_scaled) / dt
        
        print(f"   Dataset loaded: {len(self.df)} samples")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.
        
        Returns:
            Tuple of (input, target) where:
                - input: concatenated [scaled_state, scaled_action]
                - target: scaled_derivative (ds/dt)
        """
        input_tensor = torch.from_numpy(self.inputs[idx])
        target_tensor = torch.from_numpy(self.targets[idx])
        
        return input_tensor, target_tensor


# ============================================================================
# SURROGATE MODEL ARCHITECTURE
# ============================================================================

class SurrogateModel(nn.Module):
    """Neural ODE surrogate model for state transition prediction.
    
    Input: [state, action] (scaled)
    Output: d(state)/dt (scaled derivative)
    """
    
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden_layers: List[int] = None, activation: str = 'relu'):
        """Initialize surrogate model.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'elu')
        """
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = SURROGATE_CONFIG['hidden_layers']
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim
        self.output_dim = state_dim
        
        # Select activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch_size, state_dim + action_dim]
            
        Returns:
            Predicted scaled derivative d(state)/dt [batch_size, state_dim]
        """
        return self.network(x)


# ============================================================================
# DISSIPATIVE JACOBIAN REGULARIZATION
# ============================================================================

def compute_dissipative_loss(model: nn.Module, s: torch.Tensor, a: torch.Tensor, beta: float) -> torch.Tensor:
    """Computes the dissipative loss to ensure A-stability of the ODE.
    
    Enforces that the largest eigenvalue of the symmetric part of the Jacobian
    is non-positive.
    
    Args:
        model: The surrogate model
        s: Scaled state tensor [batch_size, state_dim]
        a: Scaled action tensor [batch_size, action_dim]
        beta: Regularization weight
        
    Returns:
        Scalar loss tensor
    """
    s.requires_grad_(True)
    x = torch.cat([s, a], dim=-1)
    f = model(x)  # [B, n]
    
    B, n = s.shape
    # Compute Jacobian J = df/ds
    J = torch.zeros(B, n, n, device=s.device)
    for i in range(n):
        v = torch.zeros_like(f)
        v[:, i] = 1.0
        # retain_graph=True needed if we iterate
        grad_s = torch.autograd.grad(f, s, grad_outputs=v, create_graph=True, retain_graph=True)[0]
        J[:, i, :] = grad_s
        
    # Symmetric part S = 0.5 * (J + J^T)
    S = 0.5 * (J + J.transpose(1, 2))
    
    # Power iteration to find the maximum eigenvalue of S
    # Initialize with random vectors
    v_pi = torch.randn_like(s)
    for _ in range(20):
        # bmm: [B, n, n] x [B, n, 1] -> [B, n, 1]
        Sv = torch.bmm(S, v_pi.unsqueeze(-1)).squeeze(-1)
        v_pi = torch.nn.functional.normalize(Sv, dim=-1)
        
    # Rayleigh quotient
    lam_max = (v_pi * torch.bmm(S, v_pi.unsqueeze(-1)).squeeze(-1)).sum(dim=-1)
    
    # Penalize positive eigenvalues
    penalty = torch.relu(lam_max) ** 2
    return beta * penalty.mean()


# ============================================================================
# TRAINING
# ============================================================================

def train_surrogate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    logger: ExperimentLogger = None,
    device: str = 'cpu'
) -> Tuple[nn.Module, dict]:
    """Train surrogate model with physics-informed regularization.
    
    Args:
        model: Surrogate model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        logger: Experiment logger
        device: Device to train on
        
    Returns:
        Tuple of (best_model, training_history)
    """
    model = model.to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    beta_diss = config.get('beta_diss', 0.01)
    
    print(f"\n🚀 Starting training...")
    print(f"   Device: {device}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Dissipative Beta: {beta_diss}")
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        diss_loss_total = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            s = inputs[:, :STATE_DIM]
            a = inputs[:, STATE_DIM:]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_mse = criterion(outputs, targets)
            
            # Compute dissipative loss on batch and random points
            s_rand = torch.randn_like(s)  # Scaled states roughly N(0,1)
            a_rand = torch.rand_like(a) * 2 - 1  # Random scaled actions
            
            s_combined = torch.cat([s.detach(), s_rand], dim=0)
            a_combined = torch.cat([a.detach(), a_rand], dim=0)
            
            loss_diss = compute_dissipative_loss(model, s_combined, a_combined, beta_diss)
            
            loss = loss_mse + loss_diss
            loss.backward()
            optimizer.step()
            
            train_loss += loss_mse.item()
            diss_loss_total += loss_diss.item()
        
        train_loss /= len(train_loader)
        diss_loss_total /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Log metrics
        if logger is not None:
            logger.log_metrics({
                'train_loss_mse': train_loss,
                'train_loss_diss': diss_loss_total,
                'val_loss': val_loss
            }, step=epoch)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{config['epochs']} | "
                  f"Train MSE: {train_loss:.6f} | "
                  f"Train Diss: {diss_loss_total:.6f} | "
                  f"Val MSE: {val_loss:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            history['best_epoch'] = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n⚠ Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    print(f"\n✓ Training complete!")
    print(f"   Best epoch: {history['best_epoch']+1}")
    print(f"   Best val loss: {best_val_loss:.6f}")
    
    return model, history


# ============================================================================
# ENSEMBLE UTILITIES
# ============================================================================

def get_ensemble_uncertainty(
    state: np.ndarray,
    action: np.ndarray,
    ensemble_models: List[nn.Module],
    state_scaler,
    action_scaler,
    device: str = 'cpu'
) -> float:
    """Calculate ensemble uncertainty for a state-action pair.
    
    Uncertainty is defined as the mean variance across all state dimensions
    in the normalized derivative space (ds/dt).
    
    Args:
        state: Current state [X, S, V]
        action: Action [feed_rate]
        ensemble_models: List of trained ensemble models
        state_scaler: Fitted state scaler
        action_scaler: Fitted action scaler
        device: Device for inference
        
    Returns:
        Uncertainty score
    """
    # Scale inputs
    state_scaled = state_scaler.transform(state.reshape(1, -1)).flatten()
    action_scaled = action_scaler.transform(action.reshape(1, -1)).flatten()
    input_tensor = torch.FloatTensor(np.concatenate([state_scaled, action_scaled]))
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    # Get predictions from all models (predicting ds/dt)
    predictions_scaled = []
    for model in ensemble_models:
        model.eval()
        with torch.no_grad():
            pred_scaled = model(input_tensor).cpu().numpy()
            predictions_scaled.append(pred_scaled.flatten())
    
    predictions_scaled = np.array(predictions_scaled)  # [n_models, state_dim]
    
    # Calculate variance across models per dimension
    variances = np.var(predictions_scaled, axis=0)
    
    # Mean variance across all state dimensions
    uncertainty = np.mean(variances)
    
    return uncertainty


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train surrogate model for state transition prediction'
    )
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--state_scaler_path', type=str, required=True)
    parser.add_argument('--action_scaler_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--version', type=int, required=True)
    parser.add_argument('--ensemble_index', type=int, default=None,
                       help='If provided, train as ensemble member with bootstrap')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'])
    parser.add_argument('--log_experiment', action='store_true')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SURROGATE MODEL TRAINING (NEURAL ODE)")
    print("=" * 70)
    
    # Configuration
    config = SURROGATE_CONFIG.copy()
    config['beta_diss'] = 0.01  # Default Dissipative penalty weight
    
    # Initialize logger
    logger = None
    if args.log_experiment:
        exp_name = f"surrogate_v{args.version}"
        if args.ensemble_index is not None:
            exp_name += f"_ens_{args.ensemble_index}"
        logger = ExperimentLogger(experiment_name=exp_name, config=config)
    
    # Load dataset
    print(f"\n📂 Loading dataset...")
    df = load_dataset(Path(args.dataset_path))
    
    # Split by episode if possible
    if 'episode_id' in df.columns:
        print("   Splitting by episode_id to prevent data leakage...")
        episode_ids = df['episode_id'].unique()
        np.random.shuffle(episode_ids)
        
        val_split = config.get('validation_split', 0.2)
        n_val_episodes = int(len(episode_ids) * val_split)
        # Ensure at least one validation episode if possible
        if n_val_episodes == 0 and len(episode_ids) > 1:
            n_val_episodes = 1
            
        val_episodes = episode_ids[:n_val_episodes]
        train_episodes = episode_ids[n_val_episodes:]
        
        train_df = df[df['episode_id'].isin(train_episodes)].reset_index(drop=True)
        val_df = df[df['episode_id'].isin(val_episodes)].reset_index(drop=True)
        
        print(f"   Training episodes: {len(train_episodes)}")
        print(f"   Validation episodes: {len(val_episodes)}")
    else:
        print("⚠ 'episode_id' not found in dataset! Using random split (potential data leakage).")
        # Fallback to random split
        val_split = config.get('validation_split', 0.2)
        mask = np.random.rand(len(df)) < (1 - val_split)
        train_df = df[mask].reset_index(drop=True)
        val_df = df[~mask].reset_index(drop=True)

    # Create datasets
    bootstrap = args.ensemble_index is not None
    
    train_dataset = FermentationDataset(
        dataframe=train_df,
        state_scaler_path=Path(args.state_scaler_path),
        action_scaler_path=Path(args.action_scaler_path),
        bootstrap_sample=bootstrap
    )
    
    val_dataset = FermentationDataset(
        dataframe=val_df,
        state_scaler_path=Path(args.state_scaler_path),
        action_scaler_path=Path(args.action_scaler_path),
        bootstrap_sample=False # Never bootstrap validation set
    )
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Create model
    model = SurrogateModel(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_layers=config['hidden_layers'],
        activation=config['activation']
    )
    
    print(f"\n🏗️  Model architecture:")
    print(f"   Input dim: {STATE_DIM + ACTION_DIM}")
    print(f"   Hidden layers: {config['hidden_layers']}")
    print(f"   Output dim: {STATE_DIM} (d(state)/dt)")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    model, history = train_surrogate(
        model, train_loader, val_loader, config, logger, device=args.device
    )
    
    # Save model
    if args.ensemble_index is not None:
        model_path = get_surrogate_path(args.version, ensemble_index=args.ensemble_index)
    else:
        model_path = get_surrogate_path(args.version)
    
    save_model(model, model_path)
    
    if logger is not None:
        logger.log_model(model_path)
        logger.finish()
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved: {model_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
