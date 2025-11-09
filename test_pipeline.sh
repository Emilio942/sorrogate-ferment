#!/bin/bash
# Quick test of the complete SURROGATE-FERMENT pipeline

set -e  # Exit on error

PYTHON="/home/emilio/Documents/ai/sorrogate-ferment/.venv/bin/python3.12"
cd /home/emilio/Documents/ai/sorrogate-ferment/SURROGATE-FERMENT

echo "=========================================="
echo "SURROGATE-FERMENT Pipeline Test"
echo "=========================================="

# Step 1: Generate data
echo ""
echo "Step 1/5: Generating dataset (10 episodes)..."
$PYTHON src/data_builder.py \
    --output_path data/test_pipeline.pkl \
    --n_episodes 10 \
    --policy random

# Step 2: Fit scalers
echo ""
echo "Step 2/5: Fitting scalers..."
$PYTHON src/fit_scaler.py \
    --dataset_path data/test_pipeline.pkl \
    --version test \
    --output_dir models/

# Step 3: Train surrogate
echo ""
echo "Step 3/5: Training surrogate model..."
$PYTHON src/surrogate_model.py \
    --dataset_path data/test_pipeline.pkl \
    --state_scaler_path models/scaler_state_vtest.pkl \
    --action_scaler_path models/scaler_action_vtest.pkl \
    --output_dir models/ \
    --version test \
    --device cpu

# Step 4: Train RL policy
echo ""
echo "Step 4/5: Training RL policy..."
$PYTHON src/train_rl.py \
    --surrogate_path models/surrogate_vtest_best.pth \
    --state_scaler_path models/scaler_state_vtest.pkl \
    --action_scaler_path models/scaler_action_vtest.pkl \
    --save_path models/policy_test.zip \
    --total_timesteps 2000

# Step 5: Evaluate policy
echo ""
echo "Step 5/5: Evaluating policy on HF model..."
$PYTHON src/evaluate_policy.py \
    --policy_path models/policy_test.zip \
    --policy_type rl \
    --n_episodes 3

echo ""
echo "=========================================="
echo "✅ PIPELINE TEST COMPLETE!"
echo "=========================================="
echo ""
echo "Generated files:"
ls -lh data/test_pipeline.pkl models/*test* 2>/dev/null || echo "  (check above for any missing files)"
