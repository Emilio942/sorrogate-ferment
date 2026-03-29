"""
Utility Functions for Data I/O and Budget Tracking
"""
import json
import pickle
from pathlib import Path
from typing import Any, Dict
import numpy as np


# ============================================================================
# DATA I/O FUNCTIONS
# ============================================================================

def save_dataset(data: Dict[str, np.ndarray], filepath: Path):
    """Save dataset (dictionary of arrays) to pickle file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"✓ Dataset saved to {filepath}")


def load_dataset(filepath: Path) -> Dict[str, np.ndarray]:
    """Load dataset from pickle file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"✓ Dataset loaded from {filepath}")
    return data


def save_model(model: Any, filepath: Path):
    """Save PyTorch model state dict."""
    import torch
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"✓ Model saved to {filepath}")


def load_model(model: Any, filepath: Path):
    """Load PyTorch model state dict."""
    import torch
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"✓ Model loaded from {filepath}")
    return model


def save_scaler(scaler: Any, filepath: Path):
    """Save sklearn scaler to pickle file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to {filepath}")


def load_scaler(filepath: Path) -> Any:
    """Load sklearn scaler from pickle file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Scaler not found: {filepath}")
    with open(filepath, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Scaler loaded from {filepath}")
    return scaler


# ============================================================================
# BUDGET TRACKER CLASS
# ============================================================================

class BudgetTracker:
    """
    Tracks HF model query budget (time-based).
    Persists state to JSON file for recovery across runs.
    """
    
    def __init__(self, budget_file: Path, max_budget_seconds: float):
        """
        Initialize budget tracker.
        
        Args:
            budget_file: Path to JSON file storing budget state
            max_budget_seconds: Maximum allowed HF model time (seconds)
        """
        self.budget_file = Path(budget_file)
        self.max_budget_seconds = max_budget_seconds
        self.spent_seconds = 0.0
        self.n_queries = 0
        
        # Load existing state if available
        if self.budget_file.exists():
            self.load()
        else:
            self.save()  # Create initial state file
    
    def check_budget(self, required_seconds: float) -> bool:
        """
        Check if budget allows for additional time.
        
        Args:
            required_seconds: Time required for next query
            
        Returns:
            True if budget allows, False otherwise
        """
        # Ensure we don't exceed max_budget
        return (self.spent_seconds + required_seconds) <= self.max_budget_seconds
    
    def update_budget(self, elapsed_seconds: float, n_queries: int = 1):
        """
        Update budget with actual time spent.
        
        Args:
            elapsed_seconds: Time spent on HF model query
            n_queries: Number of queries executed (default: 1)
        """
        self.spent_seconds += elapsed_seconds
        self.n_queries += n_queries
        self.save()
    
    def get_remaining(self) -> float:
        """Get remaining budget in seconds."""
        return max(0.0, self.max_budget_seconds - self.spent_seconds)
    
    def get_usage_percentage(self) -> float:
        """Get budget usage as percentage."""
        return (self.spent_seconds / self.max_budget_seconds) * 100.0
    
    def reset(self):
        """Reset budget to zero (use with caution!)."""
        self.spent_seconds = 0.0
        self.n_queries = 0
        self.save()
        print("⚠ Budget tracker reset to zero")
    
    def save(self):
        """Save current state to JSON file."""
        self.budget_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'spent_seconds': float(self.spent_seconds),
            'n_queries': int(self.n_queries),
            'max_budget_seconds': float(self.max_budget_seconds),
            'remaining_seconds': float(self.get_remaining()),
            'usage_percentage': float(self.get_usage_percentage()),
        }
        with open(self.budget_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load(self):
        """Load state from JSON file."""
        if not self.budget_file.exists():
            return
        
        with open(self.budget_file, 'r') as f:
            state = json.load(f)
        
        self.spent_seconds = state['spent_seconds']
        self.n_queries = state['n_queries']
        
        # Verify max budget hasn't changed
        if abs(state['max_budget_seconds'] - self.max_budget_seconds) > 1e-6:
            print(f"⚠ Warning: Max budget changed from {state['max_budget_seconds']} "
                  f"to {self.max_budget_seconds}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current budget status as dictionary."""
        return {
            'spent_seconds': self.spent_seconds,
            'spent_hours': self.spent_seconds / 3600,
            'n_queries': self.n_queries,
            'remaining_seconds': self.get_remaining(),
            'remaining_hours': self.get_remaining() / 3600,
            'usage_percentage': self.get_usage_percentage(),
            'budget_exceeded': self.spent_seconds > self.max_budget_seconds,
        }
    
    def print_status(self):
        """Print human-readable budget status."""
        status = self.get_status()
        print("\n" + "="*60)
        print("BUDGET STATUS")
        print("="*60)
        print(f"Queries executed:    {status['n_queries']}")
        print(f"Time spent:          {status['spent_hours']:.2f} / "
              f"{self.max_budget_seconds/3600:.2f} hours")
        print(f"Time remaining:      {status['remaining_hours']:.2f} hours")
        print(f"Usage:               {status['usage_percentage']:.1f}%")
        if status['budget_exceeded']:
            print("⚠ WARNING: BUDGET EXCEEDED!")
        print("="*60 + "\n")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_dir(path: Path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def count_parameters(model) -> int:
    """Count trainable parameters in PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}min"
    else:
        return f"{seconds/3600:.2f}h"


if __name__ == "__main__":
    # Test budget tracker
    from config import MAX_BUDGET_SECONDS, BUDGET_STATE_FILE
    
    tracker = BudgetTracker(BUDGET_STATE_FILE, MAX_BUDGET_SECONDS)
    tracker.print_status()
    
    # Simulate some queries
    print("Simulating 5 queries of 10 seconds each...")
    for i in range(5):
        if tracker.check_budget(10.0):
            tracker.update_budget(10.0)
            print(f"Query {i+1} completed")
        else:
            print(f"Query {i+1} would exceed budget!")
    
    tracker.print_status()
