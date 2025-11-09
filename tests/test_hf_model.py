"""
Tests for high-fidelity model (HF).
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hf_model import (
    hf_dynamics,
    simulate_episode,
    sample_initial_state,
    query_single_step
)
from config import HF_PARAMS, EPISODE_HORIZON


def test_hf_dynamics_zero_action():
    """Test dynamics with zero action (no substrate addition)."""
    state = np.array([1.0, 10.0])  # [biomass, substrate]
    action = np.array([0.0])       # No substrate addition
    
    dydt = hf_dynamics(0, state, action, HF_PARAMS)
    
    # Should have 2 derivatives
    assert len(dydt) == 2
    
    # With substrate, biomass should grow (dX/dt > 0)
    # (unless death rate dominates, but with our params it shouldn't)
    assert dydt[0] > -0.1  # Allow small negative for death
    
    # Substrate should decrease (dS/dt < 0) due to consumption
    assert dydt[1] < 0


def test_hf_dynamics_with_action():
    """Test dynamics with substrate addition."""
    state = np.array([1.0, 5.0])
    action = np.array([1.0])  # Add substrate
    
    dydt = hf_dynamics(0, state, action, HF_PARAMS)
    
    # Substrate derivative should be less negative (or positive)
    # compared to no action case
    dydt_no_action = hf_dynamics(0, state, np.array([0.0]), HF_PARAMS)
    
    # With substrate addition, dS/dt should be higher
    assert dydt[1] > dydt_no_action[1]


def test_hf_dynamics_no_nans():
    """Test that dynamics don't produce NaNs."""
    # Test various states
    states = [
        np.array([1.0, 10.0]),
        np.array([0.1, 0.1]),
        np.array([5.0, 20.0]),
    ]
    
    for state in states:
        action = np.array([0.5])
        dydt = hf_dynamics(0, state, action, HF_PARAMS)
        
        assert not np.isnan(dydt).any()
        assert not np.isinf(dydt).any()


def test_sample_initial_state():
    """Test initial state sampling."""
    # Sample multiple states
    for _ in range(10):
        state = sample_initial_state()
        
        assert len(state) == 2
        
        # Check bounds (from INITIAL_STATE_RANGES in config)
        assert 0.1 <= state[0] <= 1.0   # Biomass
        assert 5.0 <= state[1] <= 20.0  # Substrate


def test_simulate_episode_shape():
    """Test that episode simulation returns correct structure."""
    def random_policy(state):
        return np.array([np.random.uniform(0, 1)])
    
    initial_state = sample_initial_state()
    trajectory, elapsed_time = simulate_episode(
        initial_state,
        random_policy,
        horizon=10
    )
    
    # Check trajectory structure
    assert len(trajectory) <= 10  # May be shorter if terminated early
    assert elapsed_time > 0
    
    # Check each transition
    for state, action, next_state, reward in trajectory:
        assert len(state) == 2
        assert len(action) == 1
        assert len(next_state) == 2
        assert isinstance(reward, (int, float))


def test_simulate_episode_no_nans():
    """Test that episode doesn't produce NaNs."""
    def constant_policy(state):
        return np.array([0.5])
    
    initial_state = sample_initial_state()
    trajectory, elapsed_time = simulate_episode(
        initial_state,
        constant_policy,
        horizon=20
    )
    
    for state, action, next_state, reward in trajectory:
        assert not np.isnan(state).any()
        assert not np.isnan(action).any()
        assert not np.isnan(next_state).any()


def test_simulate_episode_positive_values():
    """Test that biomass and substrate remain non-negative."""
    def constant_policy(state):
        return np.array([0.5])
    
    initial_state = np.array([1.0, 10.0])
    trajectory, elapsed_time = simulate_episode(
        initial_state,
        constant_policy,
        horizon=30
    )
    
    for state, action, next_state, reward in trajectory:
        # Biomass should be non-negative
        assert next_state[0] >= 0
        # Substrate should be non-negative
        assert next_state[1] >= 0


def test_query_single_step():
    """Test single-step query."""
    state = np.array([1.0, 10.0])
    action = np.array([0.5])
    
    next_state, elapsed_time = query_single_step(state, action)
    
    # Check output structure
    assert len(next_state) == 2
    assert elapsed_time > 0
    
    # Check no NaNs
    assert not np.isnan(next_state).any()
    
    # Check non-negative
    assert next_state[0] >= 0
    assert next_state[1] >= 0


def test_episode_action_clipping():
    """Test that actions are clipped to valid range."""
    def extreme_policy(state):
        return np.array([100.0])  # Way too high
    
    initial_state = sample_initial_state()
    trajectory, elapsed_time = simulate_episode(
        initial_state,
        extreme_policy,
        horizon=5
    )
    
    # Actions should be clipped
    from config import MAX_SUBSTRATE_ADDITION
    for state, action, next_state, reward in trajectory:
        assert action[0] <= MAX_SUBSTRATE_ADDITION
