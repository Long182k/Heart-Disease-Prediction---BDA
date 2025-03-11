#!/usr/bin/env python3
"""
GPU-Accelerated Heart Disease Prediction Pipeline
Leverages NVIDIA RTX 3070 for high-performance model training
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import RAPIDS libraries for GPU acceleration
try:
    import cudf
    import cuml
    from cuml.preprocessing import StandardScaler
    from cuml.model_selection import train_test_split
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.linear_model import LogisticRegression as cuLR
    from cuml.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    GPU_AVAILABLE = True
    print("GPU acceleration enabled with RAPIDS!")
except ImportError:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier as cuRF
    from sklearn.linear_model import LogisticRegression as cuLR
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    GPU_AVAILABLE = False
    print("Warning: RAPIDS not found. Falling back to CPU computation.")

# Project directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "archive", "heart_disease.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, "visualizations")

def create_directories():
    """Create necessary directories"""
    for directory in [MODELS_DIR, VISUALIZATIONS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def load_data():
    """Load data optimized for GPU"""
    print(f"Loading data from {DATASET_PATH}...")
    try:
        # Load with cuDF if available for GPU acceleration
        if GPU_AVAILABLE:
            df = cudf.read_csv(DATASET_PATH)
        else:
            df = pd.read_csv(DATASET_PATH)
            
        # Rename target column and convert to numeric
        df = df.rename(columns={"Heart Disease Status": "target"})
        df['target'] = df['target'].map({'Yes': 1, 'No': 0})
        
        print(f"Loaded {len(df)} records with {len(df.columns)} columns.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_data(df):
    """Preprocess data with GPU acceleration"""
    print("Preprocessing data...")
    
    # Handle categorical features - encode them
    categorical_features = df.select_dtypes(['object']).columns
    
    for col in categorical_features:
        # Create dummy variable encoding
        df[col] = df[col].astype('category').cat.codes
    
    # Handle missing values with median imputation
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if GPU_AVAILABLE:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            else:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
    
    # Create feature engineering
    print("Engineering features...")
    
    # Age to cholesterol ratio
    df['Age_Cholesterol_Ratio'] = df['Age'] / (df['Cholesterol'] + 1e-5)
    
    # Heart rate and blood pressure product
    df['HR_BP_Product'] = df['Resting Heart Rate'] * df['Systolic Blood Pressure']
    
    # Pulse pressure
    df['Pulse_Pressure'] = df['Systolic Blood Pressure'] - df['Diastolic Blood Pressure']
    
    # Remove problematic values
    if GPU_AVAILABLE:
        df = df.fillna(0)  # cuDF doesn't have replace for inf values directly
    else:
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Preprocessed data shape: {X_scaled.shape if hasattr(X_scaled, 'shape') else (len(X_scaled), len(X_scaled.columns))}")
    return X_scaled, y

def balance_data(X, y):
    """Balance the dataset using GPU-accelerated techniques"""
    print("Balancing dataset...")
    
    # Get class counts
    if GPU_AVAILABLE:
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        n_pos, n_neg = int(n_pos), int(n_neg)
    else:
        n_pos = sum(y == 1)
        n_neg = sum(y == 0)
    
    print(f"Original class distribution: {n_pos} positive, {n_neg} negative")
    
    # If imbalanced, use downsampling of majority class
    if n_neg > n_pos:
        # Find indices of negative samples
        if GPU_AVAILABLE:
            neg_indices = (y == 0).nonzero()[0]
            pos_indices = (y == 1).nonzero()[0]
            
            # Randomly select same number of negative samples as positive
            np.random.seed(42)
            selected_neg_indices = np.random.choice(neg_indices.values_host, n_pos, replace=False)
            selected_indices = np.concatenate([pos_indices.values_host, selected_neg_indices])
            
            # Convert to cuDF compatible format
            X_balanced = X[selected_indices]
            y_balanced = y.iloc[selected_indices]
        else:
            neg_indices = [i for i, label in enumerate(y) if label == 0]
            pos_indices = [i for i, label in enumerate(y) if label == 1]
            
            # Randomly select same number of negative samples as positive
            np.random.seed(42)
            selected_neg_indices = np.random.choice(neg_indices, n_pos, replace=False)
            selected_indices = np.concatenate([pos_indices, selected_neg_indices])
            
            X_balanced = X[selected_indices]
            y_balanced = y.iloc[selected_indices]
    else:
        X_balanced = X
        y_balanced = y
    
    # Report balanced dataset size
    if GPU_AVAILABLE:
        n_pos_balanced = (y_balanced == 1).sum()
        n_neg_balanced = (y_balanced == 0).sum()
        n_pos_balanced = int(n_pos_balanced)
        n_neg_balanced = int(n_neg_balanced)
    else:
        n_pos_balanced = sum(y_balanced == 1)
        n_neg_balanced = sum(y_balanced == 0)
        
    print(f"Balanced class distribution: {n_pos_balanced} positive, {n_neg_balanced} negative")
    
    return X_balanced, y_balanced

def train_models(X, y):
    """Train models using GPU acceleration"""
    print("Training models on GPU...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models with GPU acceleration
    models = {
        'logistic_regression': cuLR(max_iter=100, verbose=0),
        'random_forest': cuRF(n_estimators=100, max_depth=10, max_features=0.3, n_bins=64)
    }
    
    # Add gradient boosting if on CPU (cuML doesn't have GBM yet)
    if not GPU_AVAILABLE:
        from sklearn.ensemble import GradientBoostingClassifier
        models['gradient_boosting'] = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    # Train and evaluate each model
    metrics_list = []
    training_times = []
    
    for name, model in models.items():
        print(f"\n====== Training {name.replace('_', ' ').title()} ======")
        
        # Time the training
        start_time = time.time()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Calculate training time
        train_time = time.time() - start_time
        training_times.append((name, train_time))
        print(f"Training time: {train_time:.2f} seconds")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities for ROC curve
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
            if isinstance(y_prob, np.ndarray) and y_prob.shape[1] > 1:
                y_prob = y_prob[:, 1]
            else:
                # Convert from GPU if needed
                if GPU_AVAILABLE:
                    y_prob = y_prob.to_pandas() if hasattr(y_prob, 'to_pandas') else y_prob
                    y_prob = y_prob.iloc[:, 1] if y_prob.shape[1] > 1 else y_prob
        else:
            # Some RAPIDS models don't have predict_proba
            y_prob = y_pred
        
        # Convert y_test and y_pred to numpy for metric calculations if on GPU
        if GPU_AVAILABLE:
            if hasattr(y_test, 'values_host'):
                y_test_np = y_test.values_host
            else:
                y_test_np = y_test
                
            if hasattr(y_pred, 'values_host'):
                y_pred_np = y_pred.values_host
            else:
                y_pred_np = y_pred
                
            if hasattr(y_prob, 'values_host'):
                y_prob_np = y_prob.values_host
            else:
                y_prob_np = y_prob
        else:
            y_test_np = y_test
            y_pred_np = y_pred
            y_prob_np = y_prob
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_np, y_pred_np)
        precision = precision_score(y_test_np, y_pred_np)
        recall = recall_score(y_test_np, y_pred_np)
        f1 = f1_score(y_test_np, y_pred_np)
        try:
            auc = roc_auc_score(y_test_np, y_prob_np)
        except:
            auc = accuracy  # Fallback if probabilities aren't available
        
        # Store metrics
        metrics = {
            'model_name': name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'training_time': train_time
        }
        metrics_list.append(metrics)
        
        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    return metrics_list, (X_test, y_test)

def save_and_visualize_results(metrics_list):
    """Save metrics and create visualizations"""
    print("Saving results...")
    
    # Save metrics to JSON
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    
    # Convert to regular Python types for JSON serialization
    if GPU_AVAILABLE:
        for m in metrics_list:
            for k, v in m.items():
                if isinstance(v, (cudf.Series, cudf.DataFrame)):
                    m[k] = v.to_pandas()
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_list, f, indent=2)
    
    print(f"Metrics saved to {metrics_path}")
    
    # Create model comparison visualization
    if metrics_list:
        plt.figure(figsize=(12, 8))
        
        model_names = [m['model_name'].replace('_', ' ').title() for m in metrics_list]
        metrics_to_plot = {
            'Accuracy': [m['accuracy'] for m in metrics_list],
            'AUC': [m['auc'] for m in metrics_list],
            'F1 Score': [m['f1'] for m in metrics_list],
            'Precision': [m['precision'] for m in metrics_list],
            'Recall': [m['recall'] for m in metrics_list]
        }
        
        # Create grouped bar chart
        bar_width = 0.15
        x = np.arange(len(model_names))
        
        for i, (metric_name, values) in enumerate(metrics_to_plot.items()):
            plt.bar(x + i*bar_width, values, bar_width, label=metric_name)
        
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison (GPU Accelerated)')
        plt.xticks(x + bar_width*2, model_names)
        plt.legend()
        plt.ylim(0, 1.0)
        
        # Save visualization
        plt.tight_layout()
        plot_path = os.path.join(VISUALIZATIONS_DIR, 'gpu_model_comparison.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Model comparison plot saved to {plot_path}")
        
        # Plot training times
        plt.figure(figsize=(10, 6))
        times = [m['training_time'] for m in metrics_list]
        plt.bar(model_names, times, color='orange')
        plt.xlabel('Model')
        plt.ylabel('Training Time (seconds)')
        plt.title('Model Training Times (GPU Accelerated)')
        plt.xticks(rotation=45)
        
        # Save visualization
        plt.tight_layout()
        time_plot_path = os.path.join(VISUALIZATIONS_DIR, 'gpu_training_times.png')
        plt.savefig(time_plot_path)
        plt.close()
        
        print(f"Training time plot saved to {time_plot_path}")
        
        # Print best model
        best_idx = max(range(len(metrics_list)), key=lambda i: metrics_list[i]['auc'])
        best_model = metrics_list[best_idx]
        print(f"\nBest model: {best_model['model_name']}")
        print(f"AUC: {best_model['auc']:.4f}")
        print(f"Accuracy: {best_model['accuracy']:.4f}")
        print(f"Training time: {best_model['training_time']:.2f} seconds")

def main():
    """Main function to run GPU-accelerated pipeline"""
    start_time = time.time()
    
    print("=" * 80)
    print("HEART DISEASE PREDICTION PIPELINE (GPU-ACCELERATED)")
    print("=" * 80)
    
    # Create directories
    create_directories()
    
    # Load data
    df = load_data()
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Balance data
    X_balanced, y_balanced = balance_data(X, y)
    
    # Train models
    metrics_list, (X_test, y_test) = train_models(X_balanced, y_balanced)
    
    # Save and visualize results
    save_and_visualize_results(metrics_list)
    
    # Print execution time
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {total_time:.2f} seconds")
    print("=" * 80)

if __name__ == "__main__":
    main()
