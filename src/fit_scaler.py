"""
Scaler fitting script for state and action normalization.
Fits StandardScaler for states and MinMaxScaler for actions.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from config import get_scaler_path, STATE_DIM, ACTION_DIM
from utils import load_dataset, save_scaler


# ============================================================================
# SCALER FITTING
# ============================================================================

def fit_state_scaler(dataset) -> StandardScaler:
    """Fit StandardScaler on state data.
    
    Combines both 'state' and 'next_state' columns for fitting.
    
    Args:
        dataset: DataFrame with 'state' and 'next_state' columns
        
    Returns:
        Fitted StandardScaler
    """
    print("\n📊 Fitting state scaler (StandardScaler)...")
    
    # Extract all states (current and next)
    states = []
    for state in dataset['state']:
        states.append(state)
    for next_state in dataset['next_state']:
        states.append(next_state)
    
    states_array = np.array(states)
    print(f"   State data shape: {states_array.shape}")
    print(f"   State ranges: [{states_array.min(axis=0)}, {states_array.max(axis=0)}]")
    
    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(states_array)
    
    print(f"   Mean: {scaler.mean_}")
    print(f"   Std: {np.sqrt(scaler.var_)}")
    
    return scaler


def fit_action_scaler(dataset) -> MinMaxScaler:
    """Fit MinMaxScaler on action data.
    
    Args:
        dataset: DataFrame with 'action' column
        
    Returns:
        Fitted MinMaxScaler
    """
    print("\n📊 Fitting action scaler (MinMaxScaler)...")
    
    # Extract all actions
    actions = []
    for action in dataset['action']:
        actions.append(action)
    
    actions_array = np.array(actions)
    print(f"   Action data shape: {actions_array.shape}")
    print(f"   Action ranges: [{actions_array.min(axis=0)}, {actions_array.max(axis=0)}]")
    
    # Fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(actions_array)
    
    print(f"   Min: {scaler.data_min_}")
    print(f"   Max: {scaler.data_max_}")
    
    return scaler


def verify_scaler(scaler, data_sample, scaler_name: str):
    """Verify scaler transformation.
    
    Args:
        scaler: Fitted scaler
        data_sample: Sample data to transform
        scaler_name: Name for logging
    """
    print(f"\n✓ Verifying {scaler_name}...")
    
    # Transform and inverse transform
    transformed = scaler.transform(data_sample[:5])
    inverse = scaler.inverse_transform(transformed)
    
    print(f"   Original sample: {data_sample[0]}")
    print(f"   Transformed: {transformed[0]}")
    print(f"   Inverse: {inverse[0]}")
    
    # Check reconstruction error
    reconstruction_error = np.abs(data_sample[:5] - inverse).max()
    print(f"   Max reconstruction error: {reconstruction_error:.6f}")
    
    if reconstruction_error > 1e-6:
        print(f"   ⚠ Warning: High reconstruction error!")
    else:
        print(f"   ✓ Reconstruction successful")


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fit scalers for state and action normalization'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path to input dataset (e.g., data/D_v1.pkl)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models',
        help='Output directory for scalers (default: models)'
    )
    parser.add_argument(
        '--version',
        type=int,
        required=True,
        help='Version number for scaler files (e.g., 1 for v1)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Run verification tests on fitted scalers'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SCALER FITTING")
    print("=" * 70)
    
    # Load dataset
    print(f"\n📂 Loading dataset from: {args.dataset_path}")
    dataset = load_dataset(Path(args.dataset_path))
    
    # Validate dataset structure
    required_columns = ['state', 'action', 'next_state', 'reward']
    for col in required_columns:
        if col not in dataset.columns:
            raise ValueError(f"Dataset missing required column: {col}")
    
    print(f"   Dataset shape: {dataset.shape}")
    print(f"   Columns: {list(dataset.columns)}")
    
    # Fit state scaler
    state_scaler = fit_state_scaler(dataset)
    
    # Fit action scaler
    action_scaler = fit_action_scaler(dataset)
    
    # Verify scalers if requested
    if args.verify:
        # Prepare sample data
        states_sample = np.array([dataset['state'].iloc[i] for i in range(min(10, len(dataset)))])
        actions_sample = np.array([dataset['action'].iloc[i] for i in range(min(10, len(dataset)))])
        
        verify_scaler(state_scaler, states_sample, "State Scaler")
        verify_scaler(action_scaler, actions_sample, "Action Scaler")
    
    # Save scalers
    print("\n💾 Saving scalers...")
    
    state_scaler_path = get_scaler_path('state', args.version)
    action_scaler_path = get_scaler_path('action', args.version)
    
    save_scaler(state_scaler, state_scaler_path)
    save_scaler(action_scaler, action_scaler_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ SCALER FITTING COMPLETE")
    print("=" * 70)
    print(f"State scaler: {state_scaler_path}")
    print(f"Action scaler: {action_scaler_path}")
    print(f"Version: v{args.version}")
    print("=" * 70)


if __name__ == '__main__':
    main()
