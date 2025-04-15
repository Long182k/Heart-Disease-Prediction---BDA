#!/usr/bin/env python3
"""
Hyperparameter Tuning Module for Cardiovascular Disease Prediction

This module provides functions for hyperparameter tuning of machine learning models
using cross-validation and grid search.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
import xgboost as xgb

# Import custom modules
sys.path.append("/Users/drake/Documents/UWE/BDA/Heart-Disease-Prediction---BDA")
from utils.smote_spark import apply_smote_to_spark_df, apply_smote_pandas

# Define constants
OUTPUT_DIR = "/Users/drake/Documents/UWE/BDA/Heart-Disease-Prediction---BDA/output"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

def tune_logistic_regression(train_df, folds=3):
    """
    Tune hyperparameters for Logistic Regression using cross-validation.
    
    Parameters:
    -----------
    train_df : pyspark.sql.DataFrame
        The training dataset
    folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    pyspark.ml.classification.LogisticRegressionModel
        The best model found
    dict
        Dictionary of best parameters
    """
    print("\nTuning Logistic Regression hyperparameters...")
    
    # Create base model
    lr = LogisticRegression(featuresCol="features_scaled", labelCol="cardio")
    
    # Create parameter grid
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1, 0.5, 1.0]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 0.8, 1.0]) \
        .addGrid(lr.maxIter, [50, 100, 200]) \
        .build()
    
    # Create evaluator
    evaluator = BinaryClassificationEvaluator(
        labelCol="cardio",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    
    # Create cross-validator
    cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=folds,
        seed=42
    )
    
    # Fit cross-validator
    start_time = time.time()
    cvModel = cv.fit(train_df)
    tuning_time = time.time() - start_time
    
    # Get best model
    best_model = cvModel.bestModel
    
    # Extract best parameters
    best_params = {
        "regParam": best_model.getRegParam(),
        "elasticNetParam": best_model.getElasticNetParam(),
        "maxIter": best_model._java_obj.getMaxIter()
    }
    
    print(f"Best parameters: {best_params}")
    print(f"Best AUC: {evaluator.evaluate(best_model.transform(train_df)):.4f}")
    print(f"Tuning time: {tuning_time:.2f} seconds")
    
    return best_model, best_params

def tune_random_forest(train_df, folds=3):
    """
    Tune hyperparameters for Random Forest using cross-validation.
    
    Parameters:
    -----------
    train_df : pyspark.sql.DataFrame
        The training dataset
    folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    pyspark.ml.classification.RandomForestClassificationModel
        The best model found
    dict
        Dictionary of best parameters
    """
    print("\nTuning Random Forest hyperparameters...")
    
    # Create base model
    rf = RandomForestClassifier(featuresCol="features_scaled", labelCol="cardio")
    
    # Create parameter grid
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [50, 100, 200]) \
        .addGrid(rf.maxDepth, [5, 10, 15, 20]) \
        .addGrid(rf.minInstancesPerNode, [1, 2, 4]) \
        .addGrid(rf.featureSubsetStrategy, ["auto", "sqrt", "log2"]) \
        .build()
    
    # Create evaluator
    evaluator = BinaryClassificationEvaluator(
        labelCol="cardio",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    
    # Create cross-validator
    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=folds,
        seed=42
    )
    
    # Fit cross-validator
    start_time = time.time()
    cvModel = cv.fit(train_df)
    tuning_time = time.time() - start_time
    
    # Get best model
    best_model = cvModel.bestModel
    
    # Extract best parameters
    best_params = {
        "numTrees": best_model._java_obj.getNumTrees(),
        "maxDepth": best_model._java_obj.getMaxDepth(),
        "minInstancesPerNode": best_model._java_obj.getMinInstancesPerNode(),
        "featureSubsetStrategy": best_model._java_obj.getFeatureSubsetStrategy()
    }
    
    print(f"Best parameters: {best_params}")
    print(f"Best AUC: {evaluator.evaluate(best_model.transform(train_df)):.4f}")
    print(f"Tuning time: {tuning_time:.2f} seconds")
    
    return best_model, best_params

def tune_gbt(train_df, folds=3):
    """
    Tune hyperparameters for Gradient Boosted Trees using cross-validation.
    
    Parameters:
    -----------
    train_df : pyspark.sql.DataFrame
        The training dataset
    folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    pyspark.ml.classification.GBTClassificationModel
        The best model found
    dict
        Dictionary of best parameters
    """
    print("\nTuning Gradient Boosted Trees hyperparameters...")
    
    # Create base model
    gbt = GBTClassifier(featuresCol="features_scaled", labelCol="cardio")
    
    # Create parameter grid
    paramGrid = ParamGridBuilder() \
        .addGrid(gbt.maxIter, [50, 100, 200]) \
        .addGrid(gbt.maxDepth, [3, 5, 8]) \
        .addGrid(gbt.stepSize, [0.05, 0.1, 0.2]) \
        .addGrid(gbt.subsamplingRate, [0.8, 1.0]) \
        .build()
    
    # Create evaluator
    evaluator = BinaryClassificationEvaluator(
        labelCol="cardio",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    
    # Create cross-validator
    cv = CrossValidator(
        estimator=gbt,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=folds,
        seed=42
    )
    
    # Fit cross-validator
    start_time = time.time()
    cvModel = cv.fit(train_df)
    tuning_time = time.time() - start_time
    
    # Get best model
    best_model = cvModel.bestModel
    
    # Extract best parameters
    best_params = {
        "maxIter": best_model._java_obj.getMaxIter(),
        "maxDepth": best_model._java_obj.getMaxDepth(),
        "stepSize": best_model._java_obj.getStepSize(),
        "subsamplingRate": best_model._java_obj.getSubsamplingRate()
    }
    
    print(f"Best parameters: {best_params}")
    print(f"Best AUC: {evaluator.evaluate(best_model.transform(train_df)):.4f}")
    print(f"Tuning time: {tuning_time:.2f} seconds")
    
    return best_model, best_params

def tune_xgboost(X_train, y_train, cv=3):
    """
    Tune hyperparameters for XGBoost using cross-validation.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        The training features
    y_train : numpy.ndarray
        The training labels
    cv : int
        Number of cross-validation folds
        
    Returns:
    --------
    xgboost.XGBClassifier
        The best model found
    dict
        Dictionary of best parameters
    """
    print("\nTuning XGBoost hyperparameters...")
    
    # Create base model
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    
    # Create parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 3, 5]
    }
    
    # Create scorer
    scorer = make_scorer(roc_auc_score)
    
    # Create grid search
    grid_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=20,  # Number of parameter settings sampled
        scoring=scorer,
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit grid search
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    tuning_time = time.time() - start_time
    
    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"Best parameters: {best_params}")
    print(f"Best AUC: {grid_search.best_score_:.4f}")
    print(f"Tuning time: {tuning_time:.2f} seconds")
    
    return best_model, best_params

def save_tuned_models(models_dict):
    """
    Save tuned models to disk.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of model names and model objects
        
    Returns:
    --------
    list
        List of saved model paths
    """
    print("\nSaving tuned models...")
    
    saved_paths = []
    
    for name, model in models_dict.items():
        if "XGBoost" in name:
            # Save scikit-learn model
            model_path = os.path.join(MODELS_DIR, f"tuned_{name.lower().replace(' ', '_')}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        else:
            # Save Spark model
            model_path = os.path.join(MODELS_DIR, f"tuned_{name.lower().replace(' ', '_')}")
            model.write().overwrite().save(model_path)
        
        saved_paths.append(model_path)
        print(f"Saved {name} model to: {model_path}")
    
    return saved_paths

def main(train_df=None, X_train=None, y_train=None):
    """
    Main function to run hyperparameter tuning.
    
    Parameters:
    -----------
    train_df : pyspark.sql.DataFrame, optional
        The Spark training dataset
    X_train : numpy.ndarray, optional
        The scikit-learn training features
    y_train : numpy.ndarray, optional
        The scikit-learn training labels
        
    Returns:
    --------
    dict
        Dictionary of tuned models
    dict
        Dictionary of best parameters for each model
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    tuned_models = {}
    best_params = {}
    
    # Tune Spark models if train_df is provided
    if train_df is not None:
        # Tune Logistic Regression
        lr_model, lr_params = tune_logistic_regression(train_df)
        tuned_models["Logistic Regression"] = lr_model
        best_params["Logistic Regression"] = lr_params
        
        # Tune Random Forest
        rf_model, rf_params = tune_random_forest(train_df)
        tuned_models["Random Forest"] = rf_model
        best_params["Random Forest"] = rf_params
        
        # Tune Gradient Boosted Trees
        gbt_model, gbt_params = tune_gbt(train_df)
        tuned_models["Gradient Boosted Trees"] = gbt_model
        best_params["Gradient Boosted Trees"] = gbt_params
    
    # Tune XGBoost if X_train and y_train are provided
    if X_train is not None and y_train is not None:
        # Tune XGBoost
        xgb_model, xgb_params = tune_xgboost(X_train, y_train)
        tuned_models["XGBoost"] = xgb_model
        best_params["XGBoost"] = xgb_params
    
    # Save tuned models
    save_tuned_models(tuned_models)
    
    return tuned_models, best_params

if __name__ == "__main__":
    print("This module should be imported and used in the main pipeline.")
    print("Run cardio_prediction_main.py to execute the complete pipeline.")