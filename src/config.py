"""
Central Configuration for SURROGATE-FERMENT Project
All hyperparameters, paths, and settings in one place.
"""
import os
import numpy as np
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# HIGH-FIDELITY MODEL PARAMETERS (Monod Kinetics)
# ============================================================================
HF_PARAMS = {
    'MU_MAX': 0.4,      # Maximum specific growth rate [1/h]
    'K_S': 0.1,         # Saturation constant [g/L]
    'YXS': 0.5,         # Yield coefficient biomass/substrate [g/g]
    'K_D': 0.01,        # Death rate constant [1/h]
    'DT': 0.1,          # Time step for integration [h]
    'HORIZON': 50,      # Episode length [steps]
}

# State space bounds
STATE_BOUNDS = {
    'biomass': (0.01, 10.0),     # [g/L]
    'substrate': (0.0, 50.0),     # [g/L]
}

# Initial state ranges for sampling
INITIAL_STATE_RANGES = STATE_BOUNDS

# Action space bounds
ACTION_BOUNDS = {
    'feed_rate': (0.0, 5.0),      # [g/h] substrate feed rate
}

# Action space arrays for Gym environments
ACTION_SPACE_LOW = np.array([ACTION_BOUNDS['feed_rate'][0]])
ACTION_SPACE_HIGH = np.array([ACTION_BOUNDS['feed_rate'][1]])

# State space arrays
STATE_SPACE_LOW = np.array([STATE_BOUNDS['biomass'][0], STATE_BOUNDS['substrate'][0]])
STATE_SPACE_HIGH = np.array([STATE_BOUNDS['biomass'][1], STATE_BOUNDS['substrate'][1]])

# Dimensions
STATE_DIM = len(STATE_BOUNDS)  # 2
ACTION_DIM = len(ACTION_BOUNDS)  # 1

# Action limits for data generation
MAX_SUBSTRATE_ADDITION = ACTION_BOUNDS['feed_rate'][1]
EPISODE_HORIZON = HF_PARAMS['HORIZON']

# ============================================================================
# BUDGET MANAGEMENT
# ============================================================================
MAX_BUDGET_SECONDS = 8 * 3600  # 8 hours total HF model usage
INITIAL_DATASET_EPISODES = 200  # Initial exploration episodes
BUDGET_STATE_FILE = DATA_DIR / "budget_state.json"

# ============================================================================
# SURROGATE MODEL CONFIGURATION
# ============================================================================
SURROGATE_CONFIG = {
    'hidden_dims': [256, 256, 128],  # MLP architecture
    'hidden_layers': [256, 256, 128],  # Alias for compatibility
    'activation': 'relu',
    'dropout': 0.1,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'batch_size': 256,
    'epochs': 100,
    'early_stopping_patience': 15,
    'val_split': 0.2,
    'validation_split': 0.2,  # Alias for compatibility
}

# Ensemble configuration
ENSEMBLE_SIZE = 5  # Number of models in ensemble
BOOTSTRAP_FRACTION = 0.8  # Fraction of data for each bootstrap sample

# ============================================================================
# REINFORCEMENT LEARNING CONFIGURATION
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
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'total_timesteps': 100000,
    'eval_freq': 1000,
    'n_eval_episodes': 10,
}

# ============================================================================
# REWARD FUNCTION (Multi-Objective)
# ============================================================================
REWARD_WEIGHTS = {
    'biomass_weight': 1.0,          # Maximize final biomass
    'substrate_cost': -0.5,         # Minimize substrate usage
    'action_penalty': -0.01,        # Penalize large actions
    'stability_bonus': 0.1,         # Reward stable trajectories
}

# ============================================================================
# ACTIVE LEARNING CONFIGURATION
# ============================================================================
AL_CONFIG = {
    'n_iterations': 10,             # Number of AL iterations
    'n_queries_per_iteration': 50,  # Queries to HF model per iteration
    'n_candidates': 1000,           # Candidate pool size
    'query_strategy': 'uncertainty', # 'uncertainty' or 'random'
    'uncertainty_method': 'ensemble_variance',
}

# ============================================================================
# DATA SCALING
# ============================================================================
SCALER_CONFIG = {
    'state_scaler_type': 'StandardScaler',  # or 'MinMaxScaler'
    'action_scaler_type': 'MinMaxScaler',
}

# ============================================================================
# LOGGING & EXPERIMENT TRACKING
# ============================================================================
LOGGING_CONFIG = {
    'use_wandb': False,             # Set to True to enable WandB
    'wandb_project': 'surrogate-ferment',
    'wandb_entity': None,           # Your WandB username
    'tensorboard_log_dir': 'runs',
    'save_frequency': 10,           # Save models every N iterations
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_dataset_path(version: str, suffix: str = "initial") -> Path:
    """Get standardized dataset path."""
    return DATA_DIR / f"D_v{version}_{suffix}.pkl"

def get_scaler_path(scaler_type: str, version: str) -> Path:
    """Get standardized scaler path."""
    return MODELS_DIR / f"scaler_{scaler_type}_v{version}.pkl"

def get_surrogate_path(version: str, ensemble_index: int = None) -> Path:
    """Get standardized surrogate model path."""
    if ensemble_index is not None:
        return MODELS_DIR / f"surrogate_v{version}_ens_{ensemble_index}.pth"
    return MODELS_DIR / f"surrogate_v{version}_best.pth"

def get_policy_path(version: str) -> Path:
    """Get standardized RL policy path."""
    return MODELS_DIR / f"policy_v{version}.zip"

def get_results_path(experiment_name: str, suffix: str = "") -> Path:
    """Get standardized results path."""
    exp_dir = RESULTS_DIR / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    if suffix:
        return exp_dir / f"{suffix}.json"
    return exp_dir

# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    """Validate configuration consistency."""
    assert HF_PARAMS['DT'] > 0, "DT must be positive"
    assert HF_PARAMS['HORIZON'] > 0, "HORIZON must be positive"
    assert MAX_BUDGET_SECONDS > 0, "Budget must be positive"
    assert INITIAL_DATASET_EPISODES > 0, "Initial episodes must be positive"
    assert ENSEMBLE_SIZE >= 1, "Ensemble size must be at least 1"
    print("✓ Configuration validated successfully")

if __name__ == "__main__":
    validate_config()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
