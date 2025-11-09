# SURROGATE-FERMENT

Active Learning for Fermentation Process Optimization using Surrogate Models and Reinforcement Learning.

## Project Overview

This project implements an active learning framework for optimizing fermentation processes. It combines:
- **High-Fidelity Model**: ODE-based fermentation model (Monod kinetics)
- **Surrogate Model**: Neural network approximation for efficient policy training
- **Reinforcement Learning**: PPO agent trained on surrogate model
- **Active Learning**: Uncertainty-based query selection for iterative improvement

## Project Structure

```
SURROGATE-FERMENT/
├── src/                    # Source code
│   ├── config.py          # Central configuration
│   ├── utils.py           # Data I/O and budget tracking
│   ├── logger.py          # Experiment logging (WandB/TensorBoard)
│   ├── hf_model.py        # High-fidelity fermentation model
│   ├── data_builder.py    # Initial dataset generation
│   ├── fit_scaler.py      # Scaler training
│   ├── surrogate_model.py # Surrogate model architecture and training
│   ├── surrogate_env.py   # Gym environment wrapper
│   ├── train_rl.py        # RL policy training
│   ├── evaluate_policy.py # Policy evaluation on HF model
│   ├── active_learning.py # Active learning loop
│   ├── main_experiment.py # Main orchestration script
│   └── visualize_results.py # Result visualization
├── data/                   # Datasets (generated)
├── models/                 # Trained models (generated)
├── results/                # Experiment results (generated)
├── plots/                  # Visualizations (generated)
├── notebooks/              # Jupyter notebooks for analysis
├── tests/                  # Unit tests
│   ├── test_config.py
│   ├── test_utils.py
│   └── test_hf_model.py
├── environment.yml         # Conda environment specification
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
cd SURROGATE-FERMENT
```

### 2. Create conda environment

```bash
conda env create -f environment.yml
conda activate surrogate-ferment
```

### 3. Verify installation

```bash
pytest tests/
```

## Quick Start

### 1. Benchmark HF Model (Optional)

Estimate how many episodes fit in the 8-hour budget:

```bash
python src/hf_model.py
```

### 2. Run Main Experiment

Execute the complete active learning loop:

```bash
python src/main_experiment.py
```

This will:
- Generate initial dataset with random policy
- Train surrogate ensemble
- Train RL policy on surrogate
- Iteratively improve via active learning
- Evaluate on HF model with budget tracking
- Log all metrics to WandB/TensorBoard

### 3. Run Baseline Evaluation

Evaluate a simple baseline policy (no budget constraint):

```bash
python src/evaluate_policy.py --policy_type baseline --use_budget False
```

### 4. Visualize Results

Generate plots comparing active learning vs baseline:

```bash
python src/visualize_results.py --results_dir results/ --output_dir plots/
```

## Configuration

Edit `src/config.py` to customize:

- **HF Model Parameters**: `MU_MAX`, `K_S`, `YXS`, `K_D`, `DT`
- **Budget**: `HF_BUDGET_SECONDS` (default: 8 hours)
- **Surrogate Hyperparameters**: `SURROGATE_CONFIG`
- **RL Hyperparameters**: `RL_CONFIG`
- **Active Learning**: `AL_CONFIG` (iterations, queries per iteration, ensemble size)
- **Reward Weights**: `REWARD_WEIGHTS` (multi-objective optimization)

## Key Features

### Budget Tracking

The `BudgetTracker` class ensures the 8-hour HF model query budget is respected:

```python
from utils import BudgetTracker

tracker = BudgetTracker()
if tracker.check_budget(estimated_time):
    # Run HF model query
    tracker.update_budget(actual_time)
```

### Experiment Logging

Automatic logging to WandB or TensorBoard:

```python
from logger import ExperimentLogger

with ExperimentLogger(experiment_name='my_run') as logger:
    logger.log_metric('reward', reward, step=iteration)
    logger.log_config(config_dict)
```

### Active Learning Loop

Iterative improvement via uncertainty-based query selection:

1. Train surrogate ensemble on current dataset
2. Train RL policy on surrogate
3. Generate candidate queries from policy rollouts
4. Select queries with highest uncertainty
5. Query HF model (with budget check)
6. Augment dataset and repeat

## Testing

Run all tests:

```bash
pytest tests/ -v
```

Run specific test file:

```bash
pytest tests/test_hf_model.py -v
```

## Development Status

**Current Phase**: Phase 1 - Foundation Complete

- [x] Project structure and configuration
- [x] High-fidelity model implementation
- [x] Budget tracking system
- [x] Experiment logging infrastructure
- [x] Basic testing framework
- [ ] Data builder (In Progress)
- [ ] Surrogate model training
- [ ] RL policy training
- [ ] Active learning implementation
- [ ] Result visualization
- [ ] Complete documentation

## References

Based on the research task list from `aufgabenliste_v3_verbessert.md`.

## License

[Add your license here]

## Contact

[Add contact information]
