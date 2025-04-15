#!/usr/bin/env python3
"""
Cardiovascular Disease Prediction - Main Script

This script orchestrates the complete pipeline for cardiovascular disease prediction:
1. Data loading and preprocessing
2. Feature engineering
3. Model training with SMOTE for class imbalance
4. Model evaluation and interpretation
5. Results visualization and reporting
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col, when, isnan, count, lit
import xgboost as xgb
import pickle
import shap

# Import custom modules
sys.path.append("/Users/drake/Documents/UWE/BDA/Heart-Disease-Prediction---BDA")
from utils.smote_spark import apply_smote_to_spark_df, apply_smote_pandas
from feature_engineering import (
    create_medical_features, encode_categorical_features, 
    create_interaction_features, normalize_features, select_features
)
from model_evaluation import (
    evaluate_spark_model, evaluate_sklearn_model, plot_confusion_matrix,
    plot_roc_curve, generate_shap_values, plot_model_comparison
)

# Define constants
DATA_PATH = "/Users/drake/Documents/UWE/BDA/Heart-Disease-Prediction---BDA/Dataset/cardio_data_processed.csv"
OUTPUT_DIR = "/Users/drake/Documents/UWE/BDA/Heart-Disease-Prediction---BDA/output"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# Create directories if they don't exist
for directory in [OUTPUT_DIR, MODELS_DIR, VISUALIZATIONS_DIR, REPORTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_spark_session(app_name="CardiovascularDiseasePredictor", memory="4g"):
    """Create and configure a Spark session optimized for the task."""
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.driver.memory", memory)
            .config("spark.executor.memory", memory)
            .config("spark.sql.shuffle.partitions", "8")  # Optimize for local execution
            .config("spark.default.parallelism", "8")
            .config("spark.driver.maxResultSize", "2g")
            .config("spark.memory.offHeap.enabled", "true")
            .config("spark.memory.offHeap.size", "2g")
            .getOrCreate())

def load_data(spark, data_path=DATA_PATH):
    """Load the cardiovascular dataset into a Spark DataFrame."""
    print(f"Loading data from {data_path}...")
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    print(f"Loaded {df.count()} records with {len(df.columns)} features")
    return df

def explore_data(df):
    """Perform exploratory data analysis on the dataset."""
    print("\n=== Dataset Overview ===")
    df.printSchema()
    
    print("\n=== Summary Statistics ===")
    summary = df.summary()
    summary.show()
    
    print("\n=== Class Distribution ===")
    class_dist = df.groupBy("cardio").count().orderBy("cardio")
    class_dist.show()
    
    # Calculate class imbalance ratio
    counts = {row["cardio"]: row["count"] for row in class_dist.collect()}
    if 0 in counts and 1 in counts:
        imbalance_ratio = counts[0] / counts[1]
        print(f"Class imbalance ratio (negative:positive): {imbalance_ratio:.2f}:1")
    
    print("\n=== Missing Values ===")
    # Count missing values in each column
    for col_name in df.columns:
        missing_count = df.filter(col(col_name).isNull() | isnan(col_name)).count()
        if missing_count > 0:
            print(f"Column {col_name}: {missing_count} missing values")
    
    # Check for class imbalance
    class_counts = df.groupBy("cardio").count().orderBy("cardio").collect()
    print("\nClass distribution:")
    for row in class_counts:
        print(f"Class {row['cardio']}: {row['count']} samples")
    
    return df

def preprocess_data(df):
    """Preprocess the data for modeling."""
    print("\n=== Preprocessing Data ===")
    
    # 1. Apply feature engineering
    print("Applying feature engineering...")
    df = create_medical_features(df)
    df = encode_categorical_features(df)
    df = create_interaction_features(df)
    
    # 2. Select features for modeling
    selected_features = select_features(df)
    print(f"Selected {len(selected_features)} features for modeling")
    
    # 3. Assemble features into a vector
    assembler = VectorAssembler(
        inputCols=selected_features,
        outputCol="features",
        handleInvalid="skip"
    )
    df = assembler.transform(df)
    
    # 4. Normalize features
    scaler = StandardScaler(
        inputCol="features",
        outputCol="features_scaled",
        withStd=True,
        withMean=True
    )
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    
    # 5. Split data into training and testing sets
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"Training set size: {train_df.count()}")
    print(f"Testing set size: {test_df.count()}")
    
    return train_df, test_df, scaler_model

def train_spark_models(train_df, test_df):
    """Train and evaluate Spark ML models."""
    print("\n=== Training Spark ML Models ===")
    
    # Check for class imbalance
    print("Checking class balance in training data...")
    class_counts = train_df.groupBy("cardio").count().orderBy("cardio").collect()
    for row in class_counts:
        print(f"Class {row['cardio']}: {row['count']} samples")
    
    # Apply SMOTE if needed
    if len(class_counts) > 1 and abs(class_counts[0]["count"] - class_counts[1]["count"]) > 0.2 * train_df.count():
        print("Applying SMOTE to handle class imbalance...")
        train_df = apply_smote_to_spark_df(train_df, features_col="features_scaled", label_col="cardio")
    
    # Define models to train
    models = {
        "Logistic Regression": LogisticRegression(
            featuresCol="features_scaled",
            labelCol="cardio",
            maxIter=100,
            regParam=0.1,
            elasticNetParam=0.8
        ),
        "Random Forest": RandomForestClassifier(
            featuresCol="features_scaled",
            labelCol="cardio",
            numTrees=100,
            maxDepth=10,
            seed=42
        ),
        "Gradient Boosted Trees": GBTClassifier(
            featuresCol="features_scaled",
            labelCol="cardio",
            maxIter=50,
            maxDepth=5,
            seed=42
        )
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        trained_model = model.fit(train_df)
        training_time = time.time() - start_time
        
        # Evaluate model
        metrics = evaluate_spark_model(
            trained_model, 
            test_df, 
            label_col="cardio", 
            prediction_col="prediction", 
            raw_prediction_col="rawPrediction"
        )
        metrics["Training Time"] = training_time
        
        # Save model
        model_path = os.path.join(MODELS_DIR, name.replace(" ", "_").lower())
        trained_model.write().overwrite().save(model_path)
        
        # Store results
        results[name] = metrics
        print(f"{name} Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    return results, models

def train_sklearn_models(train_df, test_df):
    """Train and evaluate scikit-learn models with SHAP interpretation."""
    print("\n=== Training Scikit-learn Models with SHAP Interpretation ===")
    
    # Convert Spark DataFrames to Pandas
    X_train = train_df.select("features_scaled").toPandas()
    y_train = train_df.select("cardio").toPandas()
    X_test = test_df.select("features_scaled").toPandas()
    y_test = test_df.select("cardio").toPandas()
    
    # Extract feature arrays from Spark vectors
    X_train = np.array([vec.toArray() for vec in X_train["features_scaled"]])
    y_train = y_train["cardio"].values
    X_test = np.array([vec.toArray() for vec in X_test["features_scaled"]])
    y_test = y_test["cardio"].values
    
    # Check for class imbalance
    print("Checking class balance in training data...")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"Class {cls}: {count} samples")
    
    # Apply SMOTE if needed
    if len(unique) > 1 and abs(counts[0] - counts[1]) > 0.2 * len(y_train):
        print("Applying SMOTE to handle class imbalance...")
        X_train, y_train = apply_smote_pandas(X_train, y_train)
    
    # Get feature names
    feature_names = select_features(train_df)
    
    # Train XGBoost model
    print("\nTraining XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective="binary:logistic",
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    
    start_time = time.time()
    xgb_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluate model
    metrics = evaluate_sklearn_model(xgb_model, X_test, y_test)
    metrics["Training Time"] = training_time
    
    # Save model
    model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(xgb_model, f)
    
    # Generate SHAP values for interpretation
    print("\nGenerating SHAP values for model interpretation...")
    shap_values = generate_shap_values(xgb_model, X_test, feature_names)
    
    # Generate visualizations
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    # Confusion matrix
    cm_path = os.path.join(VISUALIZATIONS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred, cm_path)
    
    # ROC curve
    roc_path = os.path.join(VISUALIZATIONS_DIR, "roc_curve.png")
    plot_roc_curve(y_test, y_pred_proba, roc_path)
    
    print("XGBoost Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return {"XGBoost": metrics}, xgb_model, feature_names

def generate_report(spark_results, sklearn_results, feature_names):
    """Generate a comprehensive report of model performance."""
    print("\n=== Generating Performance Report ===")
    
    # Combine results
    all_results = {**spark_results, **sklearn_results}
    
    # Create comparison plot
    comparison_path = os.path.join(VISUALIZATIONS_DIR, "model_comparison.png")
    plot_model_comparison(all_results, comparison_path)
    
    # Find best model
    best_model = max(all_results.items(), key=lambda x: x[1]["AUC"])
    print(f"\nBest model: {best_model[0]} with AUC = {best_model[1]['AUC']:.4f}")
    
    # Generate HTML report
    report_path = os.path.join(REPORTS_DIR, "model_performance_report.html")
    
    with open(report_path, "w") as f:
        f.write("<html><head><title>Cardiovascular Disease Prediction - Model Performance Report</title>")
        f.write("<style>body{font-family:Arial;margin:20px;} table{border-collapse:collapse;width:100%} ")
        f.write("th,td{text-align:left;padding:8px;border:1px solid #ddd} ")
        f.write("th{background-color:#f2f2f2} img{max-width:800px}</style></head><body>")
        
        # Header
        f.write("<h1>Cardiovascular Disease Prediction - Model Performance Report</h1>")
        f.write(f"<p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Model comparison
        f.write("<h2>Model Comparison</h2>")
        f.write("<table><tr><th>Model</th><th>AUC</th><th>Accuracy</th><th>Precision</th>")
        f.write("<th>Recall</th><th>F1 Score</th><th>Training Time (s)</th></tr>")
        
        for model, metrics in all_results.items():
            f.write(f"<tr><td>{model}</td>")
            f.write(f"<td>{metrics.get('AUC', 'N/A'):.4f}</td>")
            f.write(f"<td>{metrics.get('Accuracy', 'N/A'):.4f}</td>")
            f.write(f"<td>{metrics.get('Precision', 'N/A'):.4f}</td>")
            f.write(f"<td>{metrics.get('Recall', 'N/A'):.4f}</td>")
            f.write(f"<td>{metrics.get('F1 Score', 'N/A'):.4f}</td>")
            f.write(f"<td>{metrics.get('Training Time', 'N/A'):.2f}</td></tr>")
        
        f.write("</table>")
        
        # Best model
        f.write(f"<h2>Best Model: {best_model[0]}</h2>")
        f.write(f"<p>AUC: {best_model[1]['AUC']:.4f}</p>")
        
        # Visualizations
        f.write("<h2>Visualizations</h2>")
        
        f.write("<h3>Model Comparison</h3>")
        f.write(f"<img src='../visualizations/model_comparison.png' alt='Model Comparison'>")
        
        f.write("<h3>Confusion Matrix</h3>")
        f.write(f"<img src='../visualizations/confusion_matrix.png' alt='Confusion Matrix'>")
        
        f.write("<h3>ROC Curve</h3>")
        f.write(f"<img src='../visualizations/roc_curve.png' alt='ROC Curve'>")
        
        f.write("<h3>Feature Importance</h3>")
        f.write(f"<img src='../visualizations/shap_feature_importance.png' alt='Feature Importance'>")
        
        f.write("<h3>SHAP Summary Plot</h3>")
        f.write(f"<img src='../visualizations/shap_summary.png' alt='SHAP Summary'>")
        
        # Feature importance
        f.write("<h2>Feature Importance</h2>")
        f.write("<p>Top features and their impact on predictions:</p>")
        f.write("<ul>")
        for i in range(min(10, len(feature_names))):
            f.write(f"<li>{feature_names[i]}</li>")
        f.write("</ul>")
        
        f.write("</body></html>")
    
    print(f"Report generated at: {report_path}")
    return report_path

def main():
    """Main function to run the cardiovascular disease prediction pipeline."""
    print("=== Cardiovascular Disease Prediction Pipeline ===")
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        # Load and explore data
        df = load_data(spark)
        df = explore_data(df)
        
        # Preprocess data
        train_df, test_df, scaler_model = preprocess_data(df)
        
        # Train Spark ML models
        spark_results, spark_models = train_spark_models(train_df, test_df)
        
        # Train scikit-learn models with SHAP interpretation
        sklearn_results, xgb_model, feature_names = train_sklearn_models(train_df, test_df)
        
        # Generate comprehensive report
        report_path = generate_report(spark_results, sklearn_results, feature_names)
        
        print("\n=== Pipeline Completed Successfully ===")
        print(f"Results saved to: {OUTPUT_DIR}")
        print(f"Report available at: {report_path}")
        
    finally:
        # Stop Spark session
        spark.stop()

if __name__ == "__main__":
    main()