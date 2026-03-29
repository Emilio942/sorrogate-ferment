"""
Tests for utility functions and BudgetTracker.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils import (
    save_dataset, load_dataset,
    save_scaler, load_scaler,
    BudgetTracker,
    format_time
)


def _make_dummy_dataset(n_samples: int = 50, state_dim: int = 3, action_dim: int = 1):
    """Create an in-memory dummy dataset for testing."""
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame({
        'state': [rng.random(state_dim) for _ in range(n_samples)],
        'action': [rng.random(action_dim) for _ in range(n_samples)],
        'next_state': [rng.random(state_dim) for _ in range(n_samples)],
        'reward': rng.normal(size=n_samples).astype(float),
    })


def test_create_dummy_dataset():
    """Test dummy dataset creation."""
    dataset = _make_dummy_dataset(n_samples=50, state_dim=3, action_dim=1)
    
    assert len(dataset) == 50
    assert 'state' in dataset.columns
    assert 'action' in dataset.columns
    assert 'next_state' in dataset.columns
    assert 'reward' in dataset.columns
    
    # Check dimensions
    assert len(dataset['state'].iloc[0]) == 3
    assert len(dataset['action'].iloc[0]) == 1
    assert len(dataset['next_state'].iloc[0]) == 3


def test_dataset_save_load():
    """Test dataset save and load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy dataset
        dataset = _make_dummy_dataset(n_samples=20)
        
        # Save
        filepath = Path(tmpdir) / 'test_dataset.pkl'
        save_dataset(dataset, filepath)
        assert filepath.exists()
        
        # Load
        loaded_dataset = load_dataset(filepath)
        
        # Compare
        assert len(loaded_dataset) == len(dataset)
        assert list(loaded_dataset.columns) == list(dataset.columns)


def test_scaler_save_load():
    """Test scaler save and load."""
    from sklearn.preprocessing import StandardScaler
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and fit scaler
        scaler = StandardScaler()
        data = np.random.randn(100, 2)
        scaler.fit(data)
        
        # Save
        filepath = Path(tmpdir) / 'test_scaler.pkl'
        save_scaler(scaler, filepath)
        assert filepath.exists()
        
        # Load
        loaded_scaler = load_scaler(filepath)
        
        # Compare transforms
        test_data = np.random.randn(10, 2)
        original_transform = scaler.transform(test_data)
        loaded_transform = loaded_scaler.transform(test_data)
        
        np.testing.assert_array_almost_equal(original_transform, loaded_transform)


def test_budget_tracker_initialization():
    """Test BudgetTracker initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        budget_file = Path(tmpdir) / 'budget_state.json'
        
        # Create new tracker
        tracker = BudgetTracker(budget_file=budget_file, max_budget_seconds=1000.0)
        
        assert tracker.spent_seconds == 0.0
        assert tracker.n_queries == 0
        assert tracker.get_remaining() == pytest.approx(1000.0)
        assert tracker.get_usage_percentage() == pytest.approx(0.0)
        assert budget_file.exists()


def test_budget_tracker_update():
    """Test BudgetTracker update."""
    with tempfile.TemporaryDirectory() as tmpdir:
        budget_file = Path(tmpdir) / 'budget_state.json'
        tracker = BudgetTracker(budget_file=budget_file, max_budget_seconds=1000.0)
        
        # Update budget twice
        tracker.update_budget(100.0)
        tracker.update_budget(50.0)
        
        assert tracker.spent_seconds == pytest.approx(150.0)
        assert tracker.n_queries == 2
        assert tracker.get_remaining() == pytest.approx(850.0)
        assert tracker.get_usage_percentage() == pytest.approx(15.0)


def test_budget_tracker_check():
    """Test BudgetTracker check method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        budget_file = Path(tmpdir) / 'budget_state.json'
        tracker = BudgetTracker(budget_file=budget_file, max_budget_seconds=1000.0)
        
        # Should have enough budget
        assert tracker.check_budget(500.0) is True
        
        # Update budget close to limit
        tracker.update_budget(950.0)
        
        # Should not have enough budget for larger request
        assert tracker.check_budget(100.0) is False
        
        # Should allow small remaining usage
        assert tracker.check_budget(40.0) is True


def test_budget_tracker_persistence():
    """Test that BudgetTracker persists state across instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        budget_file = Path(tmpdir) / 'budget_state.json'
        
        # Create first tracker and update
        tracker1 = BudgetTracker(budget_file=budget_file, max_budget_seconds=1000.0)
        tracker1.update_budget(200.0)
        tracker1.update_budget(50.0)
        
        # Create second tracker (should load saved state)
        tracker2 = BudgetTracker(budget_file=budget_file, max_budget_seconds=1000.0)
        
        assert tracker2.spent_seconds == pytest.approx(250.0)
        assert tracker2.n_queries == 2
        assert tracker2.get_remaining() == pytest.approx(750.0)


def test_budget_tracker_reset():
    """Test BudgetTracker reset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        budget_file = Path(tmpdir) / 'budget_state.json'
        tracker = BudgetTracker(budget_file=budget_file, max_budget_seconds=1000.0)
        
        # Update budget
        tracker.update_budget(500.0)
        assert tracker.spent_seconds == pytest.approx(500.0)
        
        # Reset
        tracker.reset()
        assert tracker.spent_seconds == 0.0
        assert tracker.n_queries == 0
        assert tracker.get_remaining() == pytest.approx(1000.0)


def test_format_time():
    """Test time formatting."""
    assert format_time(30) == "30.00s"
    assert format_time(90) == "1.50min"
    assert format_time(3665) == "1.02h"
    assert format_time(7325) == "2.03h"
