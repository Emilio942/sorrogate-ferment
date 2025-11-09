"""
Central configuration file for SURROGATE-FERMENT project.
Contains all parameters for HF model, surrogate training, RL training, and experiments.
"""
import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# HIGH-FIDELITY MODEL PARAMETERS (Fermentation ODE)
# ============================================================================
# Monod kinetics parameters for microbial growth
HF_PARAMS = {
    'MU_MAX': 0.3,      # Maximum specific growth rate [1/h]
    'K_S': 0.1,         # Substrate saturation constant [g/L]
    'YXS': 0.5,         # Yield coefficient biomass/substrate [g/g]
    'K_D': 0.01,        # Death rate constant [1/h]
    'DT': 0.1,          # Time step [h]
}

# Initial state ranges (for sampling initial conditions)
INITIAL_STATE_RANGES = {
    'biomass': (0.1, 1.0),      # [g/L]
    'substrate': (5.0, 20.0),   # [g/L]
}

# Episode configuration
EPISODE_HORIZON = 50            # Number of time steps per episode
MAX_SUBSTRATE_ADDITION = 2.0    # Maximum substrate addition per step [g/L]

# ============================================================================
# BUDGET CONFIGURATION
# ============================================================================
HF_BUDGET_SECONDS = 8 * 3600    # 8 hours in seconds
INITIAL_DATASET_EPISODES = 100   # Number of episodes for initial dataset
BUDGET_STATE_FILE = DATA_DIR / "budget_state.json"

# ============================================================================
# SURROGATE MODEL HYPERPARAMETERS
# ============================================================================
SURROGATE_CONFIG = {
    'hidden_layers': [128, 128, 64],  # MLP architecture
    'activation': 'relu',
    'learning_rate': 1e-3,
    'batch_size': 64,
    'epochs': 100,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
}

# ============================================================================
# RL TRAINING HYPERPARAMETERS (PPO)
# ============================================================================
RL_CONFIG = {
    'algorithm': 'PPO',
    'policy': 'MlpPolicy',
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'total_timesteps': 100000,
}

# ============================================================================
# REWARD FUNCTION WEIGHTS (Multi-Objective)
# ============================================================================
REWARD_WEIGHTS = {
    'W_BIOMASS': 1.0,           # Weight for biomass production
    'W_SUBSTRATE_COST': 0.5,    # Weight for substrate consumption cost
}

# ============================================================================
# ACTIVE LEARNING CONFIGURATION
# ============================================================================
AL_CONFIG = {
    'max_iterations': 10,               # Maximum number of AL iterations
    'n_queries_per_iteration': 100,     # Number of queries per iteration
    'n_ensemble_models': 5,             # Number of models in ensemble
    'candidate_episodes': 20,           # Episodes to generate candidates
    'selection_method': 'uncertainty',  # 'uncertainty' or 'random'
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
EVAL_CONFIG = {
    'n_episodes': 20,                   # Number of episodes for evaluation
    'use_budget': True,                 # Whether to use budget tracking
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_CONFIG = {
    'backend': 'wandb',                 # 'wandb' or 'tensorboard'
    'project_name': 'surrogate-ferment',
    'log_interval': 100,                # Log every N steps
}

# ============================================================================
# STATE AND ACTION SPACE DIMENSIONS
# ============================================================================
STATE_DIM = 2       # [biomass, substrate]
ACTION_DIM = 1      # [substrate_addition]

# Action space bounds
ACTION_SPACE_LOW = 0.0
ACTION_SPACE_HIGH = MAX_SUBSTRATE_ADDITION

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_dataset_path(version: int, suffix: str = "") -> Path:
    """Get path for dataset file."""
    if suffix:
        return DATA_DIR / f"D_v{version}_{suffix}.pkl"
    return DATA_DIR / f"D_v{version}.pkl"

def get_scaler_path(scaler_type: str, version: int) -> Path:
    """Get path for scaler file.
    
    Args:
        scaler_type: 'state' or 'action'
        version: Version number
    """
    return MODELS_DIR / f"scaler_{scaler_type}_v{version}.pkl"

def get_surrogate_path(version: int, ensemble_index: int = None) -> Path:
    """Get path for surrogate model file.
    
    Args:
        version: Version number
        ensemble_index: If provided, returns path for ensemble member
    """
    if ensemble_index is not None:
        return MODELS_DIR / f"surrogate_v{version}_ens_{ensemble_index}.pth"
    return MODELS_DIR / f"surrogate_v{version}_best.pth"

def get_policy_path(version: int) -> Path:
    """Get path for RL policy file."""
    return MODELS_DIR / f"policy_v{version}.zip"

def get_results_path(name: str) -> Path:
    """Get path for results file."""
    return RESULTS_DIR / f"{name}.json"
