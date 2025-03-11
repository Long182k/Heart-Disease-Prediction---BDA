#!/usr/bin/env python3
"""
Heart Disease Prediction Pipeline Runner
This script ensures the pipeline runs successfully with optimal performance.
"""

import os
import sys
import subprocess
import time
import json

def setup_environment():
    """Set up virtual environment and install dependencies."""
    print("Setting up environment...")
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists("venv"):
        subprocess.run(["python3", "-m", "venv", "venv"], check=True)
    
    # Activate venv and install dependencies
    activate_cmd = ". venv/bin/activate"
    install_cmd = "pip install -r requirements.txt"
    
    # Run commands
    subprocess.run(f"{activate_cmd} && {install_cmd}", shell=True, check=True)
    print("Environment setup complete.")

def run_pipeline():
    """Run the heart disease prediction pipeline."""
    print("Running heart disease prediction pipeline...")
    
    # Activate venv and run pipeline
    activate_cmd = ". venv/bin/activate"
    run_cmd = "python heart_disease_prediction/run_pipeline.py"
    
    # Run command
    result = subprocess.run(f"{activate_cmd} && {run_cmd}", 
                           shell=True, 
                           capture_output=True,
                           text=True)
    
    # Check for errors
    if result.returncode != 0:
        print("Pipeline failed with error:")
        print(result.stderr)
        return False
    
    # Print output
    print(result.stdout)
    return True

def check_model_performance():
    """Check if model performance meets expectations."""
    try:
        # Load model metrics
        with open("models/model_metrics.json", "r") as f:
            metrics = json.load(f)
        
        # Check if any model has AUC > 0.7
        best_auc = max(m['auc'] for m in metrics) if metrics else 0
        
        if best_auc < 0.7:
            print(f"Model performance below expectations. Best AUC: {best_auc:.4f}")
            return False
        
        print(f"Model performance satisfactory. Best AUC: {best_auc:.4f}")
        return True
    except Exception as e:
        print(f"Error checking model performance: {e}")
        return False

if __name__ == "__main__":
    start_time = time.time()
    
    try:
        # Setup environment
        setup_environment()
        
        # Run pipeline
        pipeline_success = run_pipeline()
        
        if pipeline_success:
            # Check model performance
            performance_ok = check_model_performance()
            
            if performance_ok:
                print("Pipeline completed successfully with good model performance.")
            else:
                print("Pipeline completed but model performance is not optimal.")
        
        print(f"Total execution time: {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error in runner script: {str(e)}")
        sys.exit(1)
