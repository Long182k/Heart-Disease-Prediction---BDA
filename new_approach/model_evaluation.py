#!/usr/bin/env python3
"""
Model Evaluation and Interpretation Module

This module provides functions for evaluating machine learning models
and interpreting their predictions using SHAP values.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import shap

# Constants
OUTPUT_DIR = "/Users/drake/Documents/UWE/BDA/Heart-Disease-Prediction---BDA/output"
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

def evaluate_spark_model(model, test_df, label_col="cardio", prediction_col="prediction", raw_prediction_col="rawPrediction"):
    """
    Evaluate a Spark ML model on a test dataset.
    
    Parameters:
    -----------
    model : pyspark.ml.Model
        The trained Spark ML model
    test_df : pyspark.sql.DataFrame
        The test dataset
    label_col : str
        The name of the column containing the true labels
    prediction_col : str
        The name of the column containing the predicted labels
    raw_prediction_col : str
        The name of the column containing the raw predictions
        
    Returns:
    --------
    dict
        A dictionary containing evaluation metrics
    """
    # Make predictions
    predictions = model.transform(test_df)
    
    # Initialize evaluators
    binary_evaluator = BinaryClassificationEvaluator(
        labelCol=label_col, 
        rawPredictionCol=raw_prediction_col
    )
    
    multi_evaluator = MulticlassClassificationEvaluator(
        labelCol=label_col, 
        predictionCol=prediction_col
    )
    
    # Calculate metrics
    auc = binary_evaluator.evaluate(predictions)
    accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
    precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
    recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})
    f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
    
    # Return metrics
    return {
        "AUC": auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

def evaluate_sklearn_model(model, X_test, y_test):
    """
    Evaluate a scikit-learn model on a test dataset.
    
    Parameters:
    -----------
    model : sklearn.base.BaseEstimator
        The trained scikit-learn model
    X_test : numpy.ndarray or pandas.DataFrame
        The test features
    y_test : numpy.ndarray or pandas.Series
        The true labels
        
    Returns:
    --------
    dict
        A dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Return metrics
    return {
        "AUC": auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

def plot_confusion_matrix(y_true, y_pred, output_path=None):
    """
    Plot a confusion matrix.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        The true labels
    y_pred : numpy.ndarray
        The predicted labels
    output_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The confusion matrix plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        
    return plt.gcf()

def plot_roc_curve(y_true, y_pred_proba, output_path=None):
    """
    Plot a ROC curve.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        The true labels
    y_pred_proba : numpy.ndarray
        The predicted probabilities
    output_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The ROC curve plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        
    return plt.gcf()

def generate_shap_values(model, X, feature_names=None, output_dir=VISUALIZATIONS_DIR):
    """
    Generate and plot SHAP values for model interpretation.
    
    Parameters:
    -----------
    model : sklearn.base.BaseEstimator
        The trained model
    X : numpy.ndarray or pandas.DataFrame
        The feature matrix
    feature_names : list, optional
        List of feature names
    output_dir : str, optional
        Directory to save the plots
        
    Returns:
    --------
    numpy.ndarray
        The SHAP values
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create explainer
    if hasattr(model, 'predict_proba'):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    
    # If shap_values is a list (for multi-class), take the values for class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Plot summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary.png"))
    plt.close()
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_feature_importance.png"))
    plt.close()
    
    # Plot detailed feature impact for top features
    if feature_names is not None:
        # Get top 5 features by importance
        feature_importance = np.abs(shap_values).mean(0)
        top_indices = feature_importance.argsort()[-5:]
        
        for idx in top_indices:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(idx, shap_values, X, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_dependence_{feature_names[idx]}.png"))
            plt.close()
    
    return shap_values

def plot_model_comparison(metrics_dict, output_path=None):
    """
    Plot a comparison of multiple models based on their metrics.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with model names as keys and metric dictionaries as values
    output_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The model comparison plot
    """
    # Extract model names and metrics
    models = list(metrics_dict.keys())
    metrics_names = ["AUC", "Accuracy", "Precision", "Recall", "F1 Score"]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(len(metrics_names), 1, figsize=(10, 15))
    
    # Plot each metric
    for i, metric_name in enumerate(metrics_names):
        values = [metrics_dict[model][metric_name] for model in models]
        axes[i].bar(models, values, color='skyblue')
        axes[i].set_title(f"{metric_name} Comparison")
        axes[i].set_ylim(0, 1)
        
        # Add value labels on top of bars
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    
    return fig