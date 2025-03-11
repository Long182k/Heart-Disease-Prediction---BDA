#!/usr/bin/env python3
"""
Optimized Heart Disease Prediction Pipeline
Uses parallel processing for faster model training
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from joblib import parallel_backend
import warnings
warnings.filterwarnings('ignore')

# Set number of parallel jobs based on CPU cores (will utilize GPU via parallelization)
N_JOBS = -1

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
    """Load data from CSV file"""
    print(f"Loading data from {DATASET_PATH}...")
    try:
        df = pd.read_csv(DATASET_PATH)
        df.rename(columns={"Heart Disease Status": "target"}, inplace=True)
        df['target'] = df['target'].map({'Yes': 1, 'No': 0})
        print(f"Loaded {df.shape[0]} records with {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def engineer_features(df):
    """Create derived features"""
    print("Engineering features...")
    try:
        # Age to cholesterol ratio
        df['Age_Cholesterol_Ratio'] = df['Age'] / df['Cholesterol']
        
        # Heart rate and blood pressure product
        df['HR_BP_Product'] = df['Resting Heart Rate'] * df['Systolic Blood Pressure']
        
        # Pulse pressure
        df['Pulse_Pressure'] = df['Systolic Blood Pressure'] - df['Diastolic Blood Pressure']
        
        # BMI calculation, if height and weight are available
        if 'Height (cm)' in df.columns and 'Weight (kg)' in df.columns:
            df['BMI'] = df['Weight (kg)'] / ((df['Height (cm)']/100) ** 2)
        
        # Replace infinities with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        print("Feature engineering complete.")
        return df
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        return df

def preprocess_data(df):
    """Preprocess data for machine learning using parallel processing"""
    print("Preprocessing data...")
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Create preprocessing pipeline with parallel processing
    with parallel_backend('threading', n_jobs=N_JOBS):
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Fit and transform the data
        X_processed = preprocessor.fit_transform(X)
        
    print(f"Processed data shape: {X_processed.shape}")
    
    return X_processed, y, preprocessor

def handle_imbalance(X, y):
    """Handle class imbalance with SMOTE"""
    print("Handling class imbalance...")
    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"After SMOTE: {sum(y_resampled == 1)} positive samples, {sum(y_resampled == 0)} negative samples")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"Error applying SMOTE: {e}. Using original data.")
        return X, y

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models with parallel processing"""
    print("Training models with parallel processing...")
    
    models = {}
    metrics_list = []
    
    # Define models with tuning parameters
    model_configs = {
        'logistic_regression': {
            'model': LogisticRegression(max_iter=200, random_state=42),
            'params': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'solver': ['liblinear', 'saga'],
                'penalty': ['l1', 'l2']
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'max_features': ['sqrt', 'log2']
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.7, 1.0]
            }
        }
    }
    
    # Train each model with parallel processing
    for model_name, config in model_configs.items():
        try:
            print(f"\n====== {model_name.replace('_', ' ').title()} ======")
            start_time = time.time()
            
            # Create and train GridSearchCV with parallel processing
            with parallel_backend('threading', n_jobs=N_JOBS):
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=5,
                    scoring='roc_auc',
                    n_jobs=N_JOBS,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            training_time = time.time() - start_time
            models[model_name] = best_model
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            # Store metrics
            metrics = {
                'model_name': model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'training_time': training_time,
                'best_params': grid_search.best_params_
            }
            metrics_list.append(metrics)
            
            # Print results
            print(f"Training time: {training_time:.2f} seconds")
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"AUC: {auc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            # Save the model
            joblib.dump(best_model, os.path.join(MODELS_DIR, f"{model_name}_model.pkl"))
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
    
    return models, metrics_list

def save_results(metrics_list):
    """Save model metrics to JSON file"""
    print("Saving model metrics...")
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_list, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Identify best model
    if metrics_list:
        best_idx = max(range(len(metrics_list)), key=lambda i: metrics_list[i]['auc'])
        best_model = metrics_list[best_idx]
        print(f"\nBest model: {best_model['model_name']}")
        print(f"AUC: {best_model['auc']:.4f}")
        print(f"Accuracy: {best_model['accuracy']:.4f}")
        return best_model
    return None

def plot_model_comparison(metrics_list):
    """Plot model comparison bar chart"""
    if not metrics_list:
        return
        
    print("Creating model comparison visualization...")
    model_names = [m['model_name'].replace('_', ' ').title() for m in metrics_list]
    accuracies = [m['accuracy'] for m in metrics_list]
    aucs = [m['auc'] for m in metrics_list]
    f1_scores = [m['f1'] for m in metrics_list]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Performance metrics plot
    x = range(len(model_names))
    width = 0.25
    
    ax1.bar([i - width for i in x], accuracies, width, label='Accuracy')
    ax1.bar(x, aucs, width, label='AUC')
    ax1.bar([i + width for i in x], f1_scores, width, label='F1 Score')
    
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    
    # Training time plot
    training_times = [m.get('training_time', 0) for m in metrics_list]
    ax2.bar(model_names, training_times, color='orange')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Training Times')
    ax2.set_xticklabels(model_names, rotation=45)
    
    plt.tight_layout()
    plot_path = os.path.join(VISUALIZATIONS_DIR, 'model_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Model comparison plot saved to {plot_path}")

def main():
    """Main function to run the pipeline"""
    try:
        start_time = time.time()
        
        print("=" * 80)
        print("HEART DISEASE PREDICTION PIPELINE (OPTIMIZED PARALLEL VERSION)")
        print("=" * 80)
        
        # Create directories
        create_directories()
        
        # Load data
        df = load_data()
        if df is None:
            print("Error: Could not load data. Exiting.")
            return
        
        # Engineer features
        df = engineer_features(df)
        
        # Preprocess data
        X, y, preprocessor = preprocess_data(df)
        
        # Handle class imbalance
        X_resampled, y_resampled = handle_imbalance(X, y)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42
        )
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Testing set size: {X_test.shape[0]}")
        
        # Train and evaluate models
        models, metrics_list = train_models(X_train, y_train, X_test, y_test)
        
        # Save results
        best_model = save_results(metrics_list)
        
        # Plot model comparison
        if metrics_list:
            plot_model_comparison(metrics_list)
        
        # Calculate total execution time
        total_time = time.time() - start_time
        
        print("=" * 80)
        print(f"PIPELINE COMPLETED SUCCESSFULLY in {total_time:.2f} seconds")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error in pipeline: {e}")

if __name__ == "__main__":
    main()
