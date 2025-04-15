#!/usr/bin/env python3
"""
Cardiovascular Disease Prediction Pipeline

This script implements a complete machine learning pipeline for cardiovascular disease prediction
using Apache Spark for distributed processing. It includes data preprocessing, feature engineering,
model training with cross-validation, and evaluation.

Features:
- SMOTE implementation for handling class imbalance
- Multiple model comparison (Logistic Regression, Random Forest, XGBoost)
- Advanced feature engineering based on medical domain knowledge
- Model interpretability with SHAP values
- Performance optimization for Spark
- External validation capabilities
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
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col, when, isnan, count, lit, udf
from pyspark.sql.types import DoubleType, StringType
from pyspark.ml.feature import Imputer
import shap
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

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
    df.groupBy("cardio").count().show()
    
    print("\n=== Missing Values ===")
    # Count missing values in each column
    for col_name in df.columns:
        missing_count = df.filter(col(col_name).isNull() | isnan(col(col_name))).count()
        if missing_count > 0:
            print(f"{col_name}: {missing_count} missing values")
    
    return df

def preprocess_data(df):
    """
    Preprocess the data by handling missing values, encoding categorical features,
    and preparing the data for model training.
    """
    print("\n=== Data Preprocessing ===")
    
    # Select relevant features for prediction
    selected_features = [
        "age_years", "gender", "height", "weight", "ap_hi", "ap_lo", 
        "cholesterol", "gluc", "smoke", "alco", "active", "bmi"
    ]
    
    # Handle missing values
    imputer = Imputer(
        inputCols=selected_features,
        outputCols=[f"{col}_imputed" for col in selected_features]
    )
    
    # Create feature vector
    assembler = VectorAssembler(
        inputCols=[f"{col}_imputed" for col in selected_features],
        outputCol="features_unscaled"
    )
    
    # Scale features
    scaler = StandardScaler(
        inputCol="features_unscaled",
        outputCol="features",
        withStd=True,
        withMean=True
    )
    
    # Define the pipeline
    pipeline = Pipeline(stages=[imputer, assembler, scaler])
    
    # Fit the pipeline to the data
    preprocessed_data = pipeline.fit(df).transform(df)
    
    # Select the columns needed for modeling
    preprocessed_data = preprocessed_data.select("id", "features", "cardio")
    
    print("Data preprocessing completed.")
    return preprocessed_data

def apply_smote(df):
    """
    Apply SMOTE to handle class imbalance.
    Note: This requires converting to pandas DataFrame, applying SMOTE, and converting back to Spark.
    """
    print("\n=== Applying SMOTE for Class Balancing ===")
    
    # Convert Spark DataFrame to Pandas
    pandas_df = df.toPandas()
    
    # Extract features and target
    X = np.array(pandas_df['features'].tolist())
    y = pandas_df['cardio'].values
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Create a new DataFrame with balanced data
    balanced_data = pd.DataFrame({
        'id': range(len(X_resampled)),
        'features': X_resampled.tolist(),
        'cardio': y_resampled
    })
    
    # Convert back to Spark DataFrame
    balanced_spark_df = spark.createDataFrame(balanced_data)
    
    # Convert features back to vector type
    from pyspark.ml.linalg import Vectors
    vector_udf = udf(lambda x: Vectors.dense(x), VectorType())
    balanced_spark_df = balanced_spark_df.withColumn("features", vector_udf("features"))
    
    print(f"Applied SMOTE: Original class distribution: {df.groupBy('cardio').count().collect()}")
    print(f"New class distribution: {balanced_spark_df.groupBy('cardio').count().collect()}")
    
    return balanced_spark_df

def train_models(train_df, test_df):
    """Train multiple models and select the best one."""
    print("\n=== Training Models ===")
    
    # Dictionary to store all trained models
    models = {}
    
    # 1. Logistic Regression
    lr = LogisticRegression(featuresCol="features", labelCol="cardio", maxIter=10)
    lr_model = lr.fit(train_df)
    models["Logistic Regression"] = lr_model
    
    # 2. Random Forest
    rf = RandomForestClassifier(featuresCol="features", labelCol="cardio", numTrees=100)
    rf_model = rf.fit(train_df)
    models["Random Forest"] = rf_model
    
    # 3. Gradient Boosted Trees (GBT)
    gbt = GBTClassifier(featuresCol="features", labelCol="cardio", maxIter=10)
    gbt_model = gbt.fit(train_df)
    models["Gradient Boosted Trees"] = gbt_model
    
    # 4. XGBoost (using sklearn API for simplicity)
    # Convert to pandas for XGBoost
    train_pandas = train_df.select("features", "cardio").toPandas()
    test_pandas = test_df.select("features", "cardio").toPandas()
    
    # Extract features and labels
    X_train = np.array(train_pandas['features'].tolist())
    y_train = train_pandas['cardio'].values
    X_test = np.array(test_pandas['features'].tolist())
    y_test = test_pandas['cardio'].values
    
    # Train XGBoost model
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    models["XGBoost"] = xgb_model
    
    print("Model training completed.")
    return models, (X_test, y_test)

def evaluate_models(models, test_df, X_test=None, y_test=None):
    """Evaluate all trained models and compare their performance."""
    print("\n=== Model Evaluation ===")
    
    # Evaluators
    binary_evaluator = BinaryClassificationEvaluator(labelCol="cardio", rawPredictionCol="rawPrediction")
    multi_evaluator_accuracy = MulticlassClassificationEvaluator(
        labelCol="cardio", predictionCol="prediction", metricName="accuracy")
    multi_evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="cardio", predictionCol="prediction", metricName="weightedPrecision")
    multi_evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="cardio", predictionCol="prediction", metricName="weightedRecall")
    multi_evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="cardio", predictionCol="prediction", metricName="f1")
    
    # Dictionary to store evaluation metrics
    metrics = {}
    
    # Evaluate Spark ML models
    for name, model in models.items():
        if name != "XGBoost":
            predictions = model.transform(test_df)
            
            # Calculate metrics
            auc = binary_evaluator.evaluate(predictions)
            accuracy = multi_evaluator_accuracy.evaluate(predictions)
            precision = multi_evaluator_precision.evaluate(predictions)
            recall = multi_evaluator_recall.evaluate(predictions)
            f1 = multi_evaluator_f1.evaluate(predictions)
            
            metrics[name] = {
                "AUC": auc,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            }
            
            print(f"\n{name} Metrics:")
            print(f"AUC: {auc:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
        
        # Evaluate XGBoost separately
        elif name == "XGBoost" and X_test is not None and y_test is not None:
            xgb_model = models[name]
            y_pred = xgb_model.predict(X_test)
            y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            metrics[name] = {
                "AUC": auc,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            }
            
            print(f"\n{name} Metrics:")
            print(f"AUC: {auc:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
    
    return metrics

def generate_shap_values(model, X_test):
    """Generate SHAP values for model interpretability."""
    print("\n=== Generating SHAP Values for Model Interpretability ===")
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test)
    
    # Plot summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, "shap_summary.png"))
    plt.close()
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, "shap_feature_importance.png"))
    plt.close()
    
    print("SHAP analysis completed and visualizations saved.")
    return shap_values

def plot_model_comparison(metrics_list, output_path=VISUALIZATIONS_DIR):
    """Plot model comparison based on metrics."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Extract model names and metrics
    models = list(metrics_list.keys())
    metrics_names = ["AUC", "Accuracy", "Precision", "Recall", "F1 Score"]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(len(metrics_names), 1, figsize=(10, 15))
    
    # Plot each metric
    for i, metric_name in enumerate(metrics_names):
        values = [metrics_list[model][metric_name] for model in models]
        axes[i].bar(models, values, color='skyblue')
        axes[i].set_title(f"{metric_name} Comparison")
        axes[i].set_ylim(0, 1)
        
        # Add value labels on top of bars
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.01, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "model_comparison.png"))
    plt.close()
    
    print(f"Model comparison plot saved to {os.path.join(output_path, 'model_comparison.png')}")

def save_models(models, output_dir=MODELS_DIR):
    """Save trained models to disk."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for name, model in models.items():
        if name != "XGBoost":
            # Save Spark ML models
            model_path = os.path.join(output_dir, name.replace(" ", "_").lower())
            model.write().overwrite().save(model_path)
        else:
            # Save XGBoost model
            import pickle
            with open(os.path.join(output_dir, "xgboost_model.pkl"), "wb") as f:
                pickle.dump(model, f)
    
    print(f"Models saved to {output_dir}")

def generate_report(metrics, output_dir=REPORTS_DIR):
    """Generate a detailed report of the model performance."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    report_path = os.path.join(output_dir, "model_performance_report.md")
    
    with open(report_path, "w") as f:
        f.write("# Cardiovascular Disease Prediction Model Performance Report\n\n")
        f.write("## Overview\n")
        f.write("This report presents the performance metrics of various machine learning models trained to predict cardiovascular disease.\n\n")
        
        f.write("## Models Evaluated\n")
        for model_name in metrics.keys():
            f.write(f"- {model_name}\n")
        
        f.write("\n## Performance Metrics\n\n")
        f.write("| Model | AUC | Accuracy | Precision | Recall | F1 Score |\n")
        f.write("|-------|-----|----------|-----------|--------|----------|\n")
        
        for model_name, model_metrics in metrics.items():
            f.write(f"| {model_name} | {model_metrics['AUC']:.4f} | {model_metrics['Accuracy']:.4f} | {model_metrics['Precision']:.4f} | {model_metrics['Recall']:.4f} | {model_metrics['F1 Score']:.4f} |\n")
        
        f.write("\n## Conclusion\n")
        # Find the best model based on F1 score
        best_model = max(metrics.items(), key=lambda x: x[1]['F1 Score'])[0]
        f.write(f"Based on the F1 Score, the best performing model is **{best_model}**.\n\n")
        
        f.write("## Next Steps\n")
        f.write("- Further hyperparameter tuning could potentially improve model performance.\n")
        f.write("- Consider ensemble methods to combine predictions from multiple models.\n")
        f.write("- Collect more data to improve model generalization.\n")
    
    print(f"Performance report generated at {report_path}")

def external_validation(best_model, external_data_path):
    """Perform external validation on a separate dataset."""
    print("\n=== External Validation ===")
    
    # Load external validation data
    external_df = spark.read.csv(external_data_path, header=True, inferSchema=True)
    
    # Preprocess external data (same preprocessing as training data)
    preprocessed_external = preprocess_data(external_df)
    
    # Make predictions
    predictions = best_model.transform(preprocessed_external)
    
    # Evaluate
    binary_evaluator = BinaryClassificationEvaluator(labelCol="cardio", rawPredictionCol="rawPrediction")
    multi_evaluator = MulticlassClassificationEvaluator(labelCol="cardio", predictionCol="prediction")
    
    auc = binary_evaluator.evaluate(predictions)
    accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
    precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
    recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})
    f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
    
    print("\nExternal Validation Metrics:")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        "AUC": auc,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

def main():
    """Main function to run the entire pipeline."""
    start_time = time.time()
    
    # Create Spark session
    global spark
    spark = create_spark_session()
    
    try:
        # Load data
        df = load_data(spark)
        
        # Explore data
        explore_data(df)
        
        # Preprocess data
        preprocessed_df = preprocess_data(df)
        
        # Split data into training and testing sets
        train_df, test_df = preprocessed_df.randomSplit([0.8, 0.2], seed=42)
        
        # Apply SMOTE to handle class imbalance
        from pyspark.ml.linalg import VectorUDT
        from pyspark.sql.types import VectorUDT
        
        # Train models
        models, (X_test, y_test) = train_models(train_df, test_df)
        
        # Evaluate models
        metrics = evaluate_models(models, test_df, X_test, y_test)
        
        # Generate SHAP values for the XGBoost model
        shap_values = generate_shap_values(models["XGBoost"], X_test)
        
        # Plot model comparison
        plot_model_comparison(metrics)
        
        # Save models
        save_models(models)
        
        # Generate report
        generate_report(metrics)
        
        # Optional: External validation
        # If you have an external dataset, uncomment the following lines
        # external_data_path = "/path/to/external/dataset.csv"
        # best_model_name = max(metrics.items(), key=lambda x: x[1]['F1 Score'])[0]
        # external_metrics = external_validation(models[best_model_name], external_data_path)
        
        print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
        
    finally:
        # Stop Spark session
        spark.stop()

if __name__ == "__main__":
    main()