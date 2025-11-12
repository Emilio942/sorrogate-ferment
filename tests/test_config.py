"""
Tests for configuration module.
"""
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR,
    HF_PARAMS, SURROGATE_CONFIG, RL_CONFIG, REWARD_WEIGHTS, AL_CONFIG,
    get_dataset_path, get_scaler_path, get_surrogate_path, get_policy_path
)


def test_directories_exist():
    """Test that all configured directories exist."""
    assert DATA_DIR.exists()
    assert MODELS_DIR.exists()
    assert RESULTS_DIR.exists()
    assert PLOTS_DIR.exists()


def test_hf_params():
    """Test that HF parameters are properly configured."""
    assert 'MU_MAX' in HF_PARAMS
    assert 'K_S' in HF_PARAMS
    assert 'YXS' in HF_PARAMS
    assert 'K_D' in HF_PARAMS
    assert 'DT' in HF_PARAMS
    
    # Check reasonable values
    assert HF_PARAMS['MU_MAX'] > 0
    assert HF_PARAMS['K_S'] > 0
    assert HF_PARAMS['YXS'] > 0
    assert HF_PARAMS['DT'] > 0


def test_reward_weights():
    """Test that reward weights contain expected keys."""
    expected_keys = {
        'biomass_weight',
        'substrate_cost',
        'action_penalty',
        'stability_bonus'
    }
    assert expected_keys.issubset(REWARD_WEIGHTS.keys())


def test_path_helpers():
    """Test path helper functions."""
    # Dataset path
    ds_path = get_dataset_path("1")
    assert 'D_v1_initial.pkl' in str(ds_path)
    assert ds_path.parent == DATA_DIR
    
    # Scaler paths
    state_scaler_path = get_scaler_path('state', "1")
    assert 'scaler_state_v1.pkl' in str(state_scaler_path)
    
    action_scaler_path = get_scaler_path('action', "1")
    assert 'scaler_action_v1.pkl' in str(action_scaler_path)
    
    # Surrogate path
    surr_path = get_surrogate_path(1)
    assert 'surrogate_v1_best.pth' in str(surr_path)
    
    surr_ens_path = get_surrogate_path(1, ensemble_index=0)
    assert 'surrogate_v1_ens_0.pth' in str(surr_ens_path)
    
    # Policy path
    policy_path = get_policy_path(1)
    assert 'policy_v1.zip' in str(policy_path)
