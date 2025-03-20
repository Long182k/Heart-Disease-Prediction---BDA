import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import shap
from data_preprocessing import preprocess_data

# Create directories if they don't exist
def create_directories():
    """Create necessary directories for model outputs"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    directories = ['models', 'models/visualizations']
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
    print("Created output directories")

# Load and preprocess data
file_path = 'data/cardio_data_processed.csv'
X_train, X_test, y_train, y_test = preprocess_data(file_path)

def train_models_with_cv(X_train, y_train, X_test, y_test):
    """Train models with cross-validation for hyperparameter tuning"""
    # Get base directory for saving files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define parameter grids for each model
    param_grids = {
        'logistic_regression': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        },
        'random_forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', None]  # Add class weight parameter
        },
        'xgboost': {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.7, 0.9],
            'scale_pos_weight': [1, 3, 5]  # Add weight for positive class
        }
    }
    
    # Initialize base models
    from sklearn.calibration import CalibratedClassifierCV
    
    base_models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'random_forest': RandomForestClassifier(random_state=42),
        'xgboost': xgb.XGBClassifier(
            random_state=42,
            tree_method='gpu_hist',  # Use GPU acceleration
            gpu_id=0,                # Specify GPU device ID
            predictor='gpu_predictor', # Use GPU for prediction
            enable_categorical=True   # Enable categorical feature support
            # Removed use_label_encoder parameter as it's deprecated
        )
    }
    
    metrics_list = []
    
    for name, model in base_models.items():
        print(f"Training {name} with cross-validation...")
        
        # Track training time
        import time
        start_time = time.time()
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            model, 
            param_grids[name], 
            cv=10,  # 10-fold cross-validation (increased from 5)
            scoring='roc_auc', 
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Apply probability calibration
        calibrated_model = CalibratedClassifierCV(best_model, cv=10, method='isotonic')
        calibrated_model.fit(X_train, y_train)
        
        # Save both models
        model_path = os.path.join(base_dir, f'models/best_model_{name}.joblib')
        calibrated_model_path = os.path.join(base_dir, f'models/calibrated_model_{name}.joblib')
        
        joblib.dump(best_model, model_path)
        joblib.dump(calibrated_model, calibrated_model_path)
        
        # Use calibrated model for predictions
        y_pred = calibrated_model.predict(X_test)
        y_prob = calibrated_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # Find optimal threshold based on F1 score
        thresholds = np.arange(0.1, 0.9, 0.05)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred_threshold = (y_prob >= threshold).astype(int)
            f1_scores.append(f1_score(y_test, y_pred_threshold))
        
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        print(f"Optimal threshold for {name}: {optimal_threshold:.2f}")
        
        # Store metrics
        metrics = {
            'model_name': name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'training_time': float(training_time),
            'best_params': best_params,
            'optimal_threshold': float(optimal_threshold)
        }
        metrics_list.append(metrics)
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_path = os.path.join(base_dir, f'models/visualizations/{name}_confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        
        # SHAP values for interpretability (with additivity check disabled)
        try:
            if name == 'xgboost':
                explainer = shap.TreeExplainer(best_model)
                shap_values = explainer.shap_values(X_test)
            elif name == 'logistic_regression':
                # Use LinearExplainer without check_additivity parameter
                explainer = shap.LinearExplainer(best_model, X_train)
                shap_values = explainer.shap_values(X_test)
            else:
                explainer = shap.Explainer(best_model, X_train)
                shap_values = explainer(X_test, check_additivity=False)
                
            # Plot and save SHAP summary plot
            plt.figure(figsize=(10, 8))
            if name in ['xgboost', 'logistic_regression']:
                shap.summary_plot(shap_values, X_test, show=False)
            else:
                shap.summary_plot(shap_values.values, X_test, show=False)
            plt.title(f'SHAP Summary - {name}')
            plt.tight_layout()
            shap_path = os.path.join(base_dir, f'models/visualizations/{name}_shap_summary.png')
            plt.savefig(shap_path)
            plt.close()
            
            print(f"Generated SHAP values for {name}")
        except Exception as e:
            print(f"Error generating SHAP values for {name}: {e}")
            shap_values = None
        
        # Save model
        model_path = os.path.join(base_dir, f'models/best_model_{name}.joblib')
        joblib.dump(best_model, model_path)
        print(f"Saved {name} model to {model_path}")
        
        # Store metrics
        metrics = {
            'model_name': name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'training_time': float(training_time),
            'best_params': best_params
        }
        metrics_list.append(metrics)
    
    # Save metrics to JSON
    metrics_path = os.path.join(base_dir, 'models/model_metrics.json')
    # When writing model metrics to JSON, modify the model name to indicate calibration status
    for i, metrics in enumerate(metrics_list):
        if "optimal_threshold" in metrics:
            metrics["model_name"] = f"{metrics['model_name']}_calibrated"
        else:
            metrics["model_name"] = f"{metrics['model_name']}_uncalibrated"
            
    # Write the updated metrics to the JSON file
    with open(os.path.join(output_dir, 'model_metrics.json'), 'w') as f:
        json.dump(metrics_list, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    
    # Identify best model
    best_model_info = max(metrics_list, key=lambda x: x['auc'])
    print(f"\nBest model: {best_model_info['model_name']}")
    print(f"AUC: {best_model_info['auc']:.4f}")
    print(f"Accuracy: {best_model_info['accuracy']:.4f}")
    
    # Create performance comparison visualization
    model_names = [m['model_name'] for m in metrics_list]
    accuracies = [m['accuracy'] for m in metrics_list]
    aucs = [m['auc'] for m in metrics_list]
    f1_scores = [m['f1'] for m in metrics_list]
    training_times = [m['training_time'] for m in metrics_list]  # Use actual training times
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot model performance metrics
    x = np.arange(len(model_names))
    width = 0.25
    
    ax1.bar(x - width, accuracies, width, label='Accuracy', color='#1f77b4')
    ax1.bar(x, aucs, width, label='AUC', color='#ff7f0e')
    ax1.bar(x + width, f1_scores, width, label='F1 Score', color='#2ca02c')
    
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    
    # Use actual training times
    ax2.bar(model_names, training_times, color='#ff7f0e')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Training Times')
    
    plt.tight_layout()
    comparison_path = os.path.join(base_dir, 'models/visualizations/model_comparison.png')
    plt.savefig(comparison_path)
    plt.close()
    
    print(f"Saved model comparison visualization to {comparison_path}")
    
    return metrics_list

if __name__ == "__main__":
    # Create directories
    create_directories()
    
    # Train models with cross-validation
    metrics = train_models_with_cv(X_train, y_train, X_test, y_test)