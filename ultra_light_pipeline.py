#!/usr/bin/env python3
"""
Ultra-Light Heart Disease Prediction Pipeline
Designed to run on extremely limited resources
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

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

def load_and_preprocess_data():
    """Load and preprocess data in a single step"""
    print(f"Loading and preprocessing data from {DATASET_PATH}...")
    try:
        # Load data
        df = pd.read_csv(DATASET_PATH)
        print(f"Loaded {df.shape[0]} records with {df.shape[1]} columns.")
        
        # Rename target and convert to numeric
        df.rename(columns={"Heart Disease Status": "target"}, inplace=True)
        df['target'] = df['target'].map({'Yes': 1, 'No': 0})
        
        # Handle categorical features
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = pd.Categorical(df[col]).codes
        
        # Handle missing values
        df.fillna(df.median(numeric_only=True), inplace=True)
        
        # Create derived features
        df['Age_Cholesterol_Ratio'] = df['Age'] / (df['Cholesterol'] + 1)  # Add 1 to avoid division by zero
        df['HR_BP_Product'] = df['Resting Heart Rate'] * df['Systolic Blood Pressure']
        df['Pulse_Pressure'] = df['Systolic Blood Pressure'] - df['Diastolic Blood Pressure']
        
        # Drop any remaining problematic values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        # Split features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print(f"Preprocessed data shape: {X_scaled.shape}")
        return X_scaled, y
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)

def train_evaluate_models(X, y):
    """Train and evaluate simple models"""
    print("Training and evaluating models...")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define simple models
    models = {
        'logistic_regression': LogisticRegression(max_iter=100, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    }
    
    # Train and evaluate each model
    metrics_list = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # Store metrics
        metrics = {
            'model_name': name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        metrics_list.append(metrics)
        
        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    # Save metrics
    save_metrics(metrics_list)
    
    return metrics_list

def save_metrics(metrics_list):
    """Save model metrics to JSON file"""
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

def main():
    """Main function to run the ultra-light pipeline"""
    print("=" * 60)
    print("HEART DISEASE PREDICTION PIPELINE (ULTRA-LIGHT VERSION)")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Train and evaluate models
    metrics_list = train_evaluate_models(X, y)
    
    print("=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
