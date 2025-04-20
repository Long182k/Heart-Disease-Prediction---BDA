# Create Python virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip first
python -m pip install --upgrade pip

# Install wheel package to ensure binary packages are used
pip install wheel

# Install required packages with preference for binary wheels
pip install --prefer-binary -r requirements.txt --extra-index-url=https://pypi.nvidia.com

# Check GPU availability for XGBoost
python -c "try:
    import xgboost as xgb
    print('GPU available for XGBoost:', xgb.config_context(verbosity=2)['use_gpu'])
except ImportError:
    print('XGBoost not installed properly')
except Exception as e:
    print(f'Error checking GPU: {e}')"

# Run the training script
python model_training.py