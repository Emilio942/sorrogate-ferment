"""
Utility functions for data management and budget tracking.
Handles saving/loading of datasets, models, and scalers.
"""
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import torch
from config import BUDGET_STATE_FILE, HF_BUDGET_SECONDS


# ============================================================================
# DATA I/O FUNCTIONS
# ============================================================================

def save_dataset(data: pd.DataFrame, filepath: Path) -> None:
    """Save dataset to pickle file.
    
    Args:
        data: DataFrame with columns [state, action, next_state, reward]
        filepath: Path to save the file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data.to_pickle(filepath)
    print(f"Dataset saved to {filepath}")


def load_dataset(filepath: Path) -> pd.DataFrame:
    """Load dataset from pickle file.
    
    Args:
        filepath: Path to the dataset file
        
    Returns:
        DataFrame with columns [state, action, next_state, reward]
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    data = pd.read_pickle(filepath)
    print(f"Dataset loaded from {filepath} ({len(data)} samples)")
    return data


def save_model(model: torch.nn.Module, filepath: Path) -> None:
    """Save PyTorch model to file.
    
    Args:
        model: PyTorch model
        filepath: Path to save the model
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model: torch.nn.Module, filepath: Path, device: str = 'cpu') -> torch.nn.Module:
    """Load PyTorch model from file.
    
    Args:
        model: PyTorch model instance (architecture must match saved model)
        filepath: Path to the saved model
        device: Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        Loaded model
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")
    model.load_state_dict(torch.load(filepath, map_location=device))
    print(f"Model loaded from {filepath}")
    return model


def save_scaler(scaler: Any, filepath: Path) -> None:
    """Save sklearn scaler to pickle file.
    
    Args:
        scaler: Fitted sklearn scaler (e.g., StandardScaler, MinMaxScaler)
        filepath: Path to save the scaler
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {filepath}")


def load_scaler(filepath: Path) -> Any:
    """Load sklearn scaler from pickle file.
    
    Args:
        filepath: Path to the scaler file
        
    Returns:
        Loaded scaler
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Scaler not found: {filepath}")
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Scaler loaded from {filepath}")
    return scaler


# ============================================================================
# BUDGET TRACKER
# ============================================================================

class BudgetTracker:
    """Tracks HF model query budget (time-based).
    
    The tracker maintains the state of consumed budget across multiple runs
    by persisting the state to disk. This ensures budget tracking works
    across different scripts and restarts.
    """
    
    def __init__(self, budget_file: Path = BUDGET_STATE_FILE, 
                 total_budget: float = HF_BUDGET_SECONDS):
        """Initialize budget tracker.
        
        Args:
            budget_file: Path to budget state file
            total_budget: Total budget in seconds
        """
        self.budget_file = Path(budget_file)
        self.total_budget = total_budget
        self.budget_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing state or initialize
        if self.budget_file.exists():
            self._load_state()
        else:
            self.spent_time = 0.0
            self.spent_queries = 0
            self._save_state()
    
    def _load_state(self) -> None:
        """Load budget state from file."""
        with open(self.budget_file, 'r') as f:
            state = json.load(f)
        self.spent_time = state['spent_time']
        self.spent_queries = state['spent_queries']
        print(f"Budget loaded: {self.spent_time:.1f}s / {self.total_budget:.1f}s "
              f"({self.spent_queries} queries)")
    
    def _save_state(self) -> None:
        """Save budget state to file."""
        state = {
            'spent_time': self.spent_time,
            'spent_queries': self.spent_queries,
            'total_budget': self.total_budget,
            'remaining_time': self.get_remaining_time(),
            'remaining_percentage': self.get_remaining_percentage()
        }
        with open(self.budget_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def check_budget(self, required_time: float) -> bool:
        """Check if budget is sufficient for required time.
        
        Args:
            required_time: Required time in seconds
            
        Returns:
            True if budget is sufficient, False otherwise
        """
        remaining = self.get_remaining_time()
        is_sufficient = remaining >= required_time
        
        if not is_sufficient:
            print(f"WARNING: Budget insufficient! "
                  f"Required: {required_time:.1f}s, Remaining: {remaining:.1f}s")
        
        return is_sufficient
    
    def update_budget(self, spent_time: float, n_queries: int = 1) -> None:
        """Update budget with spent time.
        
        Args:
            spent_time: Time spent in seconds
            n_queries: Number of queries executed
        """
        self.spent_time += spent_time
        self.spent_queries += n_queries
        self._save_state()
        
        print(f"Budget updated: +{spent_time:.2f}s "
              f"(Total: {self.spent_time:.1f}s / {self.total_budget:.1f}s, "
              f"{self.get_remaining_percentage():.1f}% remaining)")
    
    def get_remaining_time(self) -> float:
        """Get remaining budget time in seconds."""
        return max(0.0, self.total_budget - self.spent_time)
    
    def get_remaining_percentage(self) -> float:
        """Get remaining budget as percentage."""
        return (self.get_remaining_time() / self.total_budget) * 100
    
    def reset(self) -> None:
        """Reset budget tracker to initial state."""
        self.spent_time = 0.0
        self.spent_queries = 0
        self._save_state()
        print("Budget tracker reset")
    
    def get_summary(self) -> Dict[str, float]:
        """Get budget summary.
        
        Returns:
            Dictionary with budget statistics
        """
        return {
            'total_budget_seconds': self.total_budget,
            'spent_time_seconds': self.spent_time,
            'remaining_time_seconds': self.get_remaining_time(),
            'spent_percentage': 100 - self.get_remaining_percentage(),
            'remaining_percentage': self.get_remaining_percentage(),
            'total_queries': self.spent_queries,
            'avg_time_per_query': self.spent_time / max(1, self.spent_queries)
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def estimate_episode_time(n_episodes: int, avg_time_per_episode: float) -> float:
    """Estimate total time for running n episodes.
    
    Args:
        n_episodes: Number of episodes
        avg_time_per_episode: Average time per episode in seconds
        
    Returns:
        Estimated total time in seconds
    """
    return n_episodes * avg_time_per_episode


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "2h 34m 56s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


# ============================================================================
# TESTING UTILITIES
# ============================================================================

def create_dummy_dataset(n_samples: int = 100, state_dim: int = 2, 
                        action_dim: int = 1) -> pd.DataFrame:
    """Create dummy dataset for testing.
    
    Args:
        n_samples: Number of samples
        state_dim: State dimension
        action_dim: Action dimension
        
    Returns:
        Dummy dataset
    """
    data = {
        'state': [np.random.rand(state_dim) for _ in range(n_samples)],
        'action': [np.random.rand(action_dim) for _ in range(n_samples)],
        'next_state': [np.random.rand(state_dim) for _ in range(n_samples)],
        'reward': np.random.randn(n_samples)
    }
    return pd.DataFrame(data)
