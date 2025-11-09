"""
Tests for utility functions and BudgetTracker.
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils import (
    save_dataset, load_dataset,
    save_scaler, load_scaler,
    BudgetTracker,
    create_dummy_dataset,
    format_time
)


def test_create_dummy_dataset():
    """Test dummy dataset creation."""
    dataset = create_dummy_dataset(n_samples=50, state_dim=2, action_dim=1)
    
    assert len(dataset) == 50
    assert 'state' in dataset.columns
    assert 'action' in dataset.columns
    assert 'next_state' in dataset.columns
    assert 'reward' in dataset.columns
    
    # Check dimensions
    assert len(dataset['state'].iloc[0]) == 2
    assert len(dataset['action'].iloc[0]) == 1
    assert len(dataset['next_state'].iloc[0]) == 2


def test_dataset_save_load():
    """Test dataset save and load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy dataset
        dataset = create_dummy_dataset(n_samples=20)
        
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
        tracker = BudgetTracker(budget_file=budget_file, total_budget=1000.0)
        
        assert tracker.spent_time == 0.0
        assert tracker.spent_queries == 0
        assert tracker.get_remaining_time() == 1000.0
        assert tracker.get_remaining_percentage() == 100.0
        assert budget_file.exists()


def test_budget_tracker_update():
    """Test BudgetTracker update."""
    with tempfile.TemporaryDirectory() as tmpdir:
        budget_file = Path(tmpdir) / 'budget_state.json'
        tracker = BudgetTracker(budget_file=budget_file, total_budget=1000.0)
        
        # Update budget
        tracker.update_budget(spent_time=100.0, n_queries=10)
        
        assert tracker.spent_time == 100.0
        assert tracker.spent_queries == 10
        assert tracker.get_remaining_time() == 900.0
        assert tracker.get_remaining_percentage() == 90.0


def test_budget_tracker_check():
    """Test BudgetTracker check method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        budget_file = Path(tmpdir) / 'budget_state.json'
        tracker = BudgetTracker(budget_file=budget_file, total_budget=1000.0)
        
        # Should have enough budget
        assert tracker.check_budget(500.0) == True
        
        # Update budget
        tracker.update_budget(spent_time=900.0, n_queries=50)
        
        # Should not have enough budget
        assert tracker.check_budget(200.0) == False
        
        # Should have enough for smaller request
        assert tracker.check_budget(50.0) == True


def test_budget_tracker_persistence():
    """Test that BudgetTracker persists state across instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        budget_file = Path(tmpdir) / 'budget_state.json'
        
        # Create first tracker and update
        tracker1 = BudgetTracker(budget_file=budget_file, total_budget=1000.0)
        tracker1.update_budget(spent_time=200.0, n_queries=20)
        
        # Create second tracker (should load saved state)
        tracker2 = BudgetTracker(budget_file=budget_file, total_budget=1000.0)
        
        assert tracker2.spent_time == 200.0
        assert tracker2.spent_queries == 20
        assert tracker2.get_remaining_time() == 800.0


def test_budget_tracker_reset():
    """Test BudgetTracker reset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        budget_file = Path(tmpdir) / 'budget_state.json'
        tracker = BudgetTracker(budget_file=budget_file, total_budget=1000.0)
        
        # Update budget
        tracker.update_budget(spent_time=500.0, n_queries=50)
        assert tracker.spent_time == 500.0
        
        # Reset
        tracker.reset()
        assert tracker.spent_time == 0.0
        assert tracker.spent_queries == 0
        assert tracker.get_remaining_time() == 1000.0


def test_format_time():
    """Test time formatting."""
    assert format_time(30) == "30s"
    assert format_time(90) == "1m 30s"
    assert format_time(3665) == "1h 1m 5s"
    assert format_time(7325) == "2h 2m 5s"
