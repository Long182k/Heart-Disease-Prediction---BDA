#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib

def load_data(data_path):
    """Load dataset from CSV file."""
    return pd.read_csv(data_path)

def plot_class_distribution(df, target_col='target', output_path='../visualizations'):
    """Plot the distribution of target classes."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=target_col, data=df)
    
    # Add counts above bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom', 
                   xytext = (0, 5), textcoords = 'offset points')
    
    plt.title('Distribution of Heart Disease Classes')
    plt.xlabel('Heart Disease Present (1) vs Absent (0)')
    plt.ylabel('Count')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'{output_path}/class_distribution.png')
    plt.close()
    print(f"Saved class distribution plot to {output_path}/class_distribution.png")

def plot_correlation_heatmap(df, output_path='../visualizations'):
    """Plot correlation heatmap of numerical features."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                cmap='coolwarm', square=True, linewidths=.5)
    
    plt.title('Correlation Heatmap of Features')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'{output_path}/correlation_heatmap.png')
    plt.close()
    print(f"Saved correlation heatmap to {output_path}/correlation_heatmap.png")

def plot_feature_distributions(df, output_path='../visualizations'):
    """Plot distributions of key features by target class."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Select numerical features
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col != 'target']
    
    # Limit to top 6 features to avoid too many plots
    if len(numerical_cols) > 6:
        numerical_cols = numerical_cols[:6]
    
    fig, axes = plt.subplots(len(numerical_cols), 1, figsize=(12, 4*len(numerical_cols)))
    
    for i, feature in enumerate(numerical_cols):
        ax = axes[i] if len(numerical_cols) > 1 else axes
        
        # Plot histogram with KDE
        sns.histplot(data=df, x=feature, hue='target', kde=True, ax=ax, palette='Set1',
                   element="step", common_norm=False, alpha=0.6)
        
        ax.set_title(f'Distribution of {feature} by Heart Disease Status')
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.legend(['No Disease (0)', 'Disease (1)'])
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'{output_path}/feature_distributions.png')
    plt.close()
    print(f"Saved feature distributions plot to {output_path}/feature_distributions.png")

def plot_age_gender_distribution(df, output_path='../visualizations'):
    """Plot age distribution by gender and target."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Check if 'age' and 'sex' columns exist
    if 'age' in df.columns and 'sex' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Map sex column if it's numeric
        if df['sex'].dtype in ['int64', 'float64']:
            df['sex_label'] = df['sex'].map({0: 'Female', 1: 'Male'})
        else:
            df['sex_label'] = df['sex']
        
        # Create boxplot
        sns.boxplot(x='sex_label', y='age', hue='target', data=df, palette='Set2')
        
        plt.title('Age Distribution by Gender and Heart Disease Status')
        plt.xlabel('Gender')
        plt.ylabel('Age')
        plt.legend(title='Heart Disease', labels=['Absent', 'Present'])
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f'{output_path}/age_gender_distribution.png')
        plt.close()
        print(f"Saved age-gender distribution plot to {output_path}/age_gender_distribution.png")

def plot_model_comparison(metrics_file, output_path='../visualizations'):
    """Plot model comparison based on metrics."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Load metrics from JSON file
    with open(metrics_file, 'r') as f:
        metrics_list = json.load(f)
    
    # Extract model names and metrics
    model_names = [metrics['model_name'] for metrics in metrics_list]
    accuracy = [metrics['accuracy'] for metrics in metrics_list]
    precision = [metrics['precision'] for metrics in metrics_list]
    recall = [metrics['recall'] for metrics in metrics_list]
    f1 = [metrics['f1'] for metrics in metrics_list]
    auc_values = [metrics['auc'] for metrics in metrics_list]
    
    # Create a dataframe for easier plotting
    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc_values
    })
    
    # Melt the dataframe for easier plotting
    melted_df = pd.melt(metrics_df, id_vars=['Model'], 
                        value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
                        var_name='Metric', value_name='Value')
    
    # Plot comparison
    plt.figure(figsize=(14, 8))
    chart = sns.barplot(x='Model', y='Value', hue='Metric', data=melted_df, palette='viridis')
    
    # Add values on top of bars
    for p in chart.patches:
        chart.annotate(f'{p.get_height():.3f}', 
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='bottom', fontsize=8, rotation=90,
                      xytext=(0, 5), textcoords='offset points')
    
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.legend(title='Metric', loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'{output_path}/model_comparison.png')
    plt.close()
    print(f"Saved model comparison plot to {output_path}/model_comparison.png")

def plot_confusion_matrices(test_data_path, model_dir, output_path='../visualizations'):
    """Plot confusion matrices for the models."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # TODO: This function would need the actual prediction data
    # Since we don't have the actual predictions here (they're in Spark),
    # we're skipping this for now
    
    print("Skipping confusion matrix plots as they require prediction data")

def create_all_visualizations():
    """Create all visualizations."""
    # Create visualization directory
    output_path = '../visualizations'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    try:
        # Load original dataset
        original_data = load_data('../Dataset/archive/heart_disease.csv')
        
        # Plot class distribution
        plot_class_distribution(original_data, output_path=output_path)
        
        # Plot correlation heatmap
        plot_correlation_heatmap(original_data, output_path=output_path)
        
        # Plot feature distributions
        plot_feature_distributions(original_data, output_path=output_path)
        
        # Plot age-gender distribution if columns exist
        plot_age_gender_distribution(original_data, output_path=output_path)
        
        # Plot model comparison if metrics file exists
        metrics_file = '../models/model_metrics.json'
        if os.path.exists(metrics_file):
            plot_model_comparison(metrics_file, output_path=output_path)
        
        # Plot confusion matrices if test data and models exist
        test_data_path = '../processed_data/test_data.csv'
        model_dir = '../models'
        if os.path.exists(test_data_path) and os.path.exists(model_dir):
            plot_confusion_matrices(test_data_path, model_dir, output_path=output_path)
        
        print("All visualizations created successfully!")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")

if __name__ == "__main__":
    create_all_visualizations()
