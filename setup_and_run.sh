#!/bin/bash

set -e  # Exit on error

echo "===== Heart Disease Prediction Pipeline Setup and Runner ====="

# Setup environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install wheel setuptools
pip install -r requirements.txt

# Run the pipeline
echo "Running heart disease prediction pipeline..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python heart_disease_prediction/run_pipeline.py

# Check model performance
if [ -f "models/model_metrics.json" ]; then
    echo "Checking model performance..."
    python -c "
import json
with open('models/model_metrics.json', 'r') as f:
    metrics = json.load(f)
best_model = max(metrics, key=lambda x: x['auc']) if metrics else None
if best_model:
    print(f'\nBest model: {best_model[\"model_name\"]}')
    print(f'AUC: {best_model[\"auc\"]:.4f}')
    print(f'Accuracy: {best_model[\"accuracy\"]:.4f}')
    print(f'Precision: {best_model[\"precision\"]:.4f}')
    print(f'Recall: {best_model[\"recall\"]:.4f}')
    print(f'F1 Score: {best_model[\"f1\"]:.4f}')
else:
    print('No model metrics found.')
"
fi

echo "===== Pipeline execution completed ====="
