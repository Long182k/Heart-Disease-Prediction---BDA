#!/usr/bin/env python3
"""
Model Training Module for Cardiovascular Disease Prediction

This module provides functions for training, saving, and loading machine learning models
for cardiovascular disease prediction.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_sklearn_model(model, X_train, y_train, X_val=None, y_val=None, param_grid=None):
    """
    Train a scikit-learn model.
    
    Parameters:
    -----------
    model : object
        Scikit-learn model
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_val : array-like or None
        Validation features
    y_val : array-like or None
        Validation labels
    param_grid : dict or None
        Parameter grid for grid search
        
    Returns:
    --------
    object
        Trained model
    """
    print(f"Training {model.__class__.__name__} model...")
    
    if param_grid is not None:
        print("Performing grid search...")
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    else:
        model.fit(X_train, y_train)
    
    # Evaluate on validation set if provided
    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        print("\nValidation metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
    
    return model

def train_spark_model(pipeline, train_df, val_df=None):
    """
    Train a Spark ML model.
    
    Parameters:
    -----------
    pipeline : pyspark.ml.Pipeline
        Spark ML pipeline
    train_df : pyspark.sql.DataFrame
        Training DataFrame
    val_df : pyspark.sql.DataFrame or None
        Validation DataFrame
        
    Returns:
    --------
    pyspark.ml.PipelineModel
        Trained pipeline model
    """
    print("Training Spark ML model...")
    
    # Train model
    model = pipeline.fit(train_df)
    
    # Evaluate on validation set if provided
    if val_df is not None:
        from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
        
        predictions = model.transform(val_df)
        
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
        
        print("\nValidation metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
    
    return model

def save_model(model, filepath):
    """
    Save a scikit-learn model to disk.
    
    Parameters:
    -----------
    model : object
        Trained scikit-learn model
    filepath : str
        Path to save the model
    """
    print(f"Saving model to {filepath}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save model
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved successfully")

def load_model(filepath):
    """
    Load a scikit-learn model from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
        
    Returns:
    --------
    object
        Loaded model
    """
    print(f"Loading model from {filepath}...")
    
    # Load model
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print("Model loaded successfully")
    return model