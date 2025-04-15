#!/usr/bin/env python3
"""
Evaluation Module for Cardiovascular Disease Prediction

This module provides functions for evaluating machine learning models
for cardiovascular disease prediction.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    classification_report
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col

def evaluate_sklearn_model(model, X_test, y_test):
    """
    Evaluate a scikit-learn model on the test set.
    
    Parameters:
    -----------
    model : object
        Trained scikit-learn model
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
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
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    
    # Return metrics
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc,
        'Confusion Matrix': cm,
        'Classification Report': report,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def evaluate_spark_model(model, test_df):
    """
    Evaluate a Spark ML model on the test set.
    
    Parameters:
    -----------
    model : object
        Trained Spark ML model
    test_df : pyspark.sql.DataFrame
        Test DataFrame
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Make predictions
    predictions = model.transform(test_df)
    
    # Binary classification evaluator for AUC
    evaluator_auc = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction",
        labelCol="cardio",
        metricName="areaUnderROC"
    )
    auc = evaluator_auc.evaluate(predictions)
    
    # Multiclass classification evaluator for accuracy, precision, recall, f1
    evaluator_multi = MulticlassClassificationEvaluator(
        labelCol="cardio",
        predictionCol="prediction"
    )
    
    accuracy = evaluator_multi.setMetricName("accuracy").evaluate(predictions)
    precision = evaluator_multi.setMetricName("weightedPrecision").evaluate(predictions)
    recall = evaluator_multi.setMetricName("weightedRecall").evaluate(predictions)
    f1 = evaluator_multi.setMetricName("f1").evaluate(predictions)
    
    # Convert predictions to Pandas for confusion matrix
    pred_pd = predictions.select("cardio", "prediction").toPandas()
    cm = confusion_matrix(pred_pd["cardio"], pred_pd["prediction"])
    
    # Return metrics
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc,
        'Confusion Matrix': cm,
        'Predictions': predictions
    }

def plot_confusion_matrix(y_true, y_pred, output_path=None):
    """
    Plot a confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    output_path : str or None
        Path to save the plot, if None, the plot is displayed
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_true, y_pred_proba, output_path=None):
    """
    Plot a ROC curve.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    output_path : str or None
        Path to save the plot, if None, the plot is displayed
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_precision_recall_curve(y_true, y_pred_proba, output_path=None):
    """
    Plot a Precision-Recall curve.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    output_path : str or None
        Path to save the plot, if None, the plot is displayed
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(model, feature_names, output_path=None):
    """
    Plot feature importance for tree-based models.
    
    Parameters:
    -----------
    model : object
        Trained tree-based model (e.g., RandomForest, GradientBoosting)
    feature_names : list
        List of feature names
    output_path : str or None
        Path to save the plot, if None, the plot is displayed
    """
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Rearrange feature names so they match the sorted feature importances
        names = [feature_names[i] for i in indices]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), names, rotation=90)
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    else:
        print("Model does not have feature_importances_ attribute")

def compare_models(results_dict, output_path=None):
    """
    Compare multiple models based on their evaluation metrics.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and evaluation results as values
    output_path : str or None
        Path to save the plot, if None, the plot is displayed
    """
    # Extract metrics for each model
    models = list(results_dict.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    
    # Create DataFrame for comparison
    comparison = pd.DataFrame({
        model_name: {
            metric: results_dict[model_name][metric]
            for metric in metrics
        } for model_name in models
    }).T
    
    # Print comparison table
    print("\nModel Comparison:")
    print(comparison)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    comparison.plot(kind='bar', figsize=(12, 8))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.grid(axis='y')
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
    
    return comparison

def generate_evaluation_report(results_dict, output_dir):
    """
    Generate a comprehensive evaluation report for multiple models.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and evaluation results as values
    output_dir : str
        Directory to save the report and visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare models
    comparison = compare_models(
        results_dict, 
        output_path=os.path.join(output_dir, 'model_comparison.png')
    )
    
    # Save comparison to CSV
    comparison.to_csv(os.path.join(output_dir, 'model_comparison.csv'))
    
    # Generate visualizations for each model
    for model_name, results in results_dict.items():
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            results['y_true'],
            results['y_pred'],
            output_path=os.path.join(model_dir, 'confusion_matrix.png')
        )
        
        # Plot ROC curve
        plot_roc_curve(
            results['y_true'],
            results['y_pred_proba'],
            output_path=os.path.join(model_dir, 'roc_curve.png')
        )
        
        # Plot Precision-Recall curve
        plot_precision_recall_curve(
            results['y_true'],
            results['y_pred_proba'],
            output_path=os.path.join(model_dir, 'precision_recall_curve.png')
        )
        
        # Save classification report
        with open(os.path.join(model_dir, 'classification_report.txt'), 'w') as f:
            f.write(results['Classification Report'])
    
    print(f"Evaluation report generated in {output_dir}")