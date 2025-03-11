#!/usr/bin/env python3
"""
PyTorch GPU-Accelerated Heart Disease Prediction Pipeline
Designed to leverage NVIDIA RTX 3070 GPU
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Project directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "archive", "heart_disease.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, "visualizations")

# Check for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def create_directories():
    """Create necessary directories"""
    for directory in [MODELS_DIR, VISUALIZATIONS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def load_and_preprocess_data():
    """Load and preprocess the heart disease dataset"""
    print(f"Loading data from {DATASET_PATH}...")
    try:
        # Load data
        df = pd.read_csv(DATASET_PATH)
        print(f"Loaded {df.shape[0]} records with {df.shape[1]} columns.")
        
        # Rename target and convert to numeric
        df.rename(columns={"Heart Disease Status": "target"}, inplace=True)
        df['target'] = df['target'].map({'Yes': 1, 'No': 0})
        
        # Feature engineering
        print("Engineering features...")
        
        # Handle missing values first
        df.fillna(df.median(numeric_only=True), inplace=True)
        
        # Create derived features
        df['Age_Cholesterol_Ratio'] = df['Age'] / (df['Cholesterol'] + 1)  # Add 1 to avoid division by zero
        df['HR_BP_Product'] = df['Resting Heart Rate'] * df['Systolic Blood Pressure']
        df['Pulse_Pressure'] = df['Systolic Blood Pressure'] - df['Diastolic Blood Pressure']
        
        # Clean up any infinities
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Preprocess features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        # Apply preprocessing
        X_processed = preprocessor.fit_transform(X)
        print(f"Processed data shape: {X_processed.shape}")
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_processed)
        y_tensor = torch.FloatTensor(y.values)
        
        return X_tensor, y_tensor, preprocessor, X_processed.shape[1]
    
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)

class NeuralNetwork(nn.Module):
    """Neural Network model for Heart Disease prediction"""
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.layer4(x))
        return x

def balance_classes(X, y):
    """Balance the dataset using undersampling"""
    # Count class instances
    neg_indices = (y == 0).nonzero(as_tuple=True)[0]
    pos_indices = (y == 1).nonzero(as_tuple=True)[0]
    
    # Get counts
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    
    print(f"Original class distribution: {n_pos} positive, {n_neg} negative")
    
    # Balance by undersampling majority class
    if n_neg > n_pos:
        # Randomly select n_pos negative samples
        np.random.seed(42)
        selected_neg_indices = np.random.choice(neg_indices.cpu().numpy(), n_pos, replace=False)
        selected_neg_indices = torch.LongTensor(selected_neg_indices)
        
        # Combine with positive samples
        selected_indices = torch.cat([pos_indices, selected_neg_indices])
        
        # Select balanced dataset
        X_balanced = X[selected_indices]
        y_balanced = y[selected_indices]
        
        print(f"Balanced class distribution: {n_pos} positive, {n_pos} negative")
    else:
        X_balanced = X
        y_balanced = y
        print("Data already balanced or positive class is majority.")
    
    return X_balanced, y_balanced

def train_neural_network(X, y, input_size):
    """Train neural network model with GPU acceleration"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Move to GPU
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Create model
    model = NeuralNetwork(input_size).to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create DataLoader for mini-batch training
    train_dataset = TensorDataset(X_train, y_train.view(-1, 1))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Training loop
    epochs = 100
    start_time = time.time()
    
    print(f"Training neural network on {device}...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
    
    training_time = time.time() - start_time
    print(f"Training complete in {training_time:.2f} seconds")
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_test)
        y_pred = (y_pred_prob > 0.5).float()
        
        # Convert to numpy for metrics calculation
        y_test_np = y_test.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy().flatten()
        y_pred_prob_np = y_pred_prob.cpu().numpy().flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_np, y_pred_np)
        precision = precision_score(y_test_np, y_pred_np)
        recall = recall_score(y_test_np, y_pred_np)
        f1 = f1_score(y_test_np, y_pred_np)
        auc = roc_auc_score(y_test_np, y_pred_prob_np)
    
    # Save metrics
    metrics = {
        'model_name': 'neural_network',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'training_time': training_time
    }
    
    print(f"Neural Network Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save model
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "neural_network.pt"))
    
    return metrics, model

def save_results(metrics):
    """Save metrics to JSON file"""
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    
    # Load existing metrics if available
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            try:
                existing_metrics = json.load(f)
                
                # Check if neural network already exists
                nn_exists = False
                for i, m in enumerate(existing_metrics):
                    if m['model_name'] == 'neural_network':
                        existing_metrics[i] = metrics
                        nn_exists = True
                        break
                
                if not nn_exists:
                    existing_metrics.append(metrics)
                
                metrics_list = existing_metrics
            except:
                metrics_list = [metrics]
    else:
        metrics_list = [metrics]
    
    # Save metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics_list, f, indent=2)
    
    print(f"Metrics saved to {metrics_path}")
    return metrics_list

def visualize_results(metrics_list):
    """Create visualizations of model performance"""
    if not metrics_list:
        return
    
    # Plot metrics comparison
    plt.figure(figsize=(12, 6))
    
    model_names = [m['model_name'].replace('_', ' ').title() for m in metrics_list]
    metrics_to_plot = {
        'Accuracy': [m['accuracy'] for m in metrics_list],
        'AUC': [m['auc'] for m in metrics_list],
        'F1 Score': [m['f1'] for m in metrics_list]
    }
    
    # Create grouped bar chart
    bar_width = 0.25
    x = np.arange(len(model_names))
    
    for i, (metric_name, values) in enumerate(metrics_to_plot.items()):
        plt.bar(x + i*bar_width, values, bar_width, label=metric_name)
    
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison (GPU Accelerated)')
    plt.xticks(x + bar_width, model_names)
    plt.legend()
    plt.ylim(0, 1.0)
    
    # Save visualization
    plt.tight_layout()
    plot_path = os.path.join(VISUALIZATIONS_DIR, 'pytorch_model_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Model comparison plot saved to {plot_path}")
    
    # Plot training times if available
    if all('training_time' in m for m in metrics_list):
        plt.figure(figsize=(10, 6))
        times = [m.get('training_time', 0) for m in metrics_list]
        plt.bar(model_names, times, color='orange')
        plt.xlabel('Model')
        plt.ylabel('Training Time (seconds)')
        plt.title('Model Training Times (GPU Accelerated)')
        plt.xticks(rotation=45)
        
        # Save visualization
        plt.tight_layout()
        time_plot_path = os.path.join(VISUALIZATIONS_DIR, 'pytorch_training_times.png')
        plt.savefig(time_plot_path)
        plt.close()
        
        print(f"Training time plot saved to {time_plot_path}")

def main():
    """Main function to run the PyTorch GPU pipeline"""
    start_time = time.time()
    
    print("=" * 80)
    print("HEART DISEASE PREDICTION PIPELINE (PYTORCH GPU-ACCELERATED)")
    print("=" * 80)
    
    # Create directories
    create_directories()
    
    # Load and preprocess data
    X, y, preprocessor, input_size = load_and_preprocess_data()
    
    # Balance classes
    X_balanced, y_balanced = balance_classes(X, y)
    
    # Train neural network
    nn_metrics, nn_model = train_neural_network(X_balanced, y_balanced, input_size)
    
    # Save and visualize results
    metrics_list = save_results(nn_metrics)
    visualize_results(metrics_list)
    
    # Calculate total execution time
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {total_time:.2f} seconds")
    print("=" * 80)

if __name__ == "__main__":
    main()
