#!/usr/bin/env python3
"""
Heart Disease Prediction Pipeline

This script combines all steps of the heart disease prediction pipeline:
1. Data preprocessing
2. Model training
3. Visualization

Run this script to execute the complete pipeline in one go.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import time
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer, StringIndexer, OneHotEncoder, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when, lit
import matplotlib.pyplot as plt
import seaborn as sns

# Define constants correctly
# Project directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "archive", "heart_disease.csv")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, "visualizations")

def create_directories():
    """Create necessary directories if they don't exist."""
    for directory in [PROCESSED_DATA_DIR, MODELS_DIR, VISUALIZATIONS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def create_spark_session(app_name="Heart Disease Prediction"):
    """Create a Spark session with optimized settings for limited resources."""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "1g") \
        .config("spark.executor.memory", "1g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "1g") \
        .config("spark.driver.maxResultSize", "512m") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.default.parallelism", "4") \
        .config("spark.memory.fraction", "0.6") \
        .config("spark.memory.storageFraction", "0.2") \
        .config("spark.rdd.compress", "true") \
        .getOrCreate()

def get_spark_session():
    """Get or create the Spark session."""
    if 'spark' not in globals():
        global spark
        spark = create_spark_session()
    return spark

# ----- DATA PREPROCESSING -----

def load_data(filepath):
    """Load data from the CSV file."""
    print(f"Loading data from {filepath}...")
    try:
        spark = get_spark_session()
        df = spark.read.csv(filepath, header=True, inferSchema=True)
        print(f"Loaded {df.count()} records with {len(df.columns)} columns.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Preprocess the data: handle missing values, convert categorical features, etc."""
    print("Preprocessing data...")
    
    # Print schema
    print("Original Schema:")
    df.printSchema()
    
    # Rename target column
    df = df.withColumnRenamed("Heart Disease Status", "target")
    
    # Convert target to numeric: "Yes" -> 1.0, "No" -> 0.0
    df = df.withColumn("target", when(df["target"] == "Yes", 1.0).otherwise(0.0))
    
    print("Converted target column to numeric values (1.0 for Yes, 0.0 for No)")
    
    # Check for missing values
    print("Checking for missing values:")
    for col_name in df.columns:
        missing_count = df.filter(df[col_name].isNull()).count()
        if missing_count > 0:
            print(f"Column {col_name} has {missing_count} missing values.")
    
    # List of numerical features
    numerical_features = [col_name for col_name in df.columns 
                         if col_name != 'target' and df.select(col_name).dtypes[0][1] != 'string']
    
    # List of categorical features
    categorical_features = [col_name for col_name in df.columns 
                           if col_name != 'target' and df.select(col_name).dtypes[0][1] == 'string']
    
    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    from pyspark.ml.feature import StringIndexer, OneHotEncoder
    
    # Feature Engineering: Create derived features
    if "Age" in df.columns and "Cholesterol" in df.columns:
        df = df.withColumn("Age_Cholesterol_Ratio", col("Age") / col("Cholesterol"))
    
    if "Resting Heart Rate" in df.columns and "Systolic Blood Pressure" in df.columns:
        df = df.withColumn("HR_BP_Product", col("Resting Heart Rate") * col("Systolic Blood Pressure"))
    
    if "Systolic Blood Pressure" in df.columns and "Diastolic Blood Pressure" in df.columns:
        df = df.withColumn("Pulse_Pressure", col("Systolic Blood Pressure") - col("Diastolic Blood Pressure"))
        
    if "Weight" in df.columns and "Height" in df.columns:
        # Calculate BMI if Weight is in kg and Height is in cm
        df = df.withColumn("BMI", col("Weight") / (col("Height") * col("Height") / 10000))
    
    # Get updated features after engineering
    numerical_features = [col_name for col_name in df.columns 
                         if col_name != 'target' and df.select(col_name).dtypes[0][1] != 'string']
    
    # Impute missing values in numerical features
    imputer = Imputer(
        inputCols=numerical_features,
        outputCols=[f"{col_name}_imputed" for col_name in numerical_features]
    ).setStrategy("mean")
    
    # Process categorical features
    # 1. String indexing (convert categories to indices)
    indexers = [StringIndexer(inputCol=col_name, 
                              outputCol=f"{col_name}_indexed",
                              handleInvalid="keep") 
               for col_name in categorical_features]
    
    # 2. One-hot encoding (convert indices to binary vectors)
    encoder = OneHotEncoder(
        inputCols=[f"{col_name}_indexed" for col_name in categorical_features],
        outputCols=[f"{col_name}_encoded" for col_name in categorical_features]
    )
    
    # Assemble features into a vector
    numerical_assembled = [f"{col_name}_imputed" for col_name in numerical_features]
    categorical_assembled = [f"{col_name}_encoded" for col_name in categorical_features]
    all_features = numerical_assembled + categorical_assembled
    
    # Create assembler for all features
    from pyspark.ml.feature import VectorAssembler
    assembler = VectorAssembler(inputCols=all_features, outputCol="features")
    
    # Create preprocessing pipeline without ChiSqSelector to reduce memory usage
    preprocessing_pipeline = Pipeline(stages=indexers + [imputer, encoder, assembler])
    
    # Fit preprocessing pipeline
    preprocessing_model = preprocessing_pipeline.fit(df)
    processed_df = preprocessing_model.transform(df)
    
    # Select relevant columns
    final_df = processed_df.select("target", "features")
    
    # Show some processed data
    print("Processed Data Sample:")
    final_df.show(5)
    
    return final_df, preprocessing_model

def handle_imbalanced_data(df, spark):
    """Handle imbalanced data using random undersampling."""
    print("Handling imbalanced data...")
    
    # Count samples in each class
    class_counts = df.groupBy("target").count().orderBy("target")
    class_counts.show()
    
    # Check if imbalanced
    class_counts_list = class_counts.collect()
    negative_count = class_counts_list[0]["count"]
    positive_count = class_counts_list[1]["count"] if len(class_counts_list) > 1 else 0
    
    if positive_count == 0:
        print("Warning: No positive samples in the dataset")
        return df
    
    # If classes are already balanced or nearly balanced, return original df
    if positive_count / negative_count > 0.8:
        print("Classes are already reasonably balanced, no resampling needed")
        return df
    
    # Use random undersampling for simplicity and to avoid memory issues
    # Sample the majority class to match the minority class
    majority_df = df.filter(df["target"] == 0.0)
    minority_df = df.filter(df["target"] == 1.0)
    
    # Calculate sampling fraction
    sampling_fraction = min(1.0, positive_count / negative_count * 2)  # At most balance 2:1
    sampled_majority = majority_df.sample(withReplacement=False, fraction=sampling_fraction, seed=42)
    
    # Combine minority and sampled majority
    balanced_df = sampled_majority.union(minority_df)
    
    # Show new class distribution
    print("Balanced class distribution:")
    balanced_df.groupBy("target").count().orderBy("target").show()
    
    return balanced_df

def split_data(df, train_ratio=0.8):
    """Split data into training and testing sets."""
    print(f"Splitting data into training ({train_ratio*100}%) and testing ({(1-train_ratio)*100}%) sets...")
    
    # Split the data
    train_df, test_df = df.randomSplit([train_ratio, 1 - train_ratio], seed=42)
    
    # Print the sizes
    print(f"Training set size: {train_df.count()}")
    print(f"Testing set size: {test_df.count()}")
    
    return train_df, test_df

def save_processed_data(train_df, test_df, output_dir=PROCESSED_DATA_DIR):
    """Save processed data for later use."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save training data
    train_path = os.path.join(output_dir, "train")
    test_path = os.path.join(output_dir, "test")
    
    # Save in Parquet format
    try:
        train_df.write.mode("overwrite").parquet(train_path)
        test_df.write.mode("overwrite").parquet(test_path)
        print(f"Saved processed data to {output_dir}/")
    except Exception as e:
        print(f"Error saving processed data: {e}")
    
    return train_path, test_path

# ----- MODEL TRAINING -----

def build_logistic_regression_model(train_df):
    """Build and train a Logistic Regression model."""
    print("Building Logistic Regression model...")
    
    # Create Logistic Regression model
    lr = LogisticRegression(
        featuresCol="features", 
        labelCol="target",
        maxIter=20,        # Reduced from 100
        regParam=0.01,
        elasticNetParam=0.0,
        standardization=True,
        threshold=0.5,
        family="binomial"
    )
    
    # Define parameter grid
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
        .build()
    
    # Create cross-validator
    evaluator = BinaryClassificationEvaluator(
        labelCol="target", 
        metricName="areaUnderROC"
    )
    
    # Create cross-validator
    cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=2,      # Reduced from 3
        parallelism=1,   # Serial execution to save memory
        seed=42
    )
    
    # Train the model
    print("Training Logistic Regression model...")
    cvModel = cv.fit(train_df)
    best_model = cvModel.bestModel
    
    print("Logistic Regression training complete.")
    
    return best_model

def build_random_forest_model(train_df):
    """Build and train a Random Forest model with reduced complexity."""
    print("Building Random Forest model...")
    
    # Create Random Forest model with modest defaults
    rf = RandomForestClassifier(
        featuresCol="features", 
        labelCol="target",
        numTrees=20,           # Reduced from 100
        maxDepth=10,           # Reduced from 20
        maxBins=32,            # Reduced from 128
        minInstancesPerNode=5, # Increased from 1
        seed=42
    )
    
    # Define simplified parameter grid for hyperparameter tuning
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()
    
    # Create cross-validator
    evaluator = BinaryClassificationEvaluator(
        labelCol="target", 
        metricName="areaUnderROC"
    )
    
    # Create cross-validator with fewer folds
    cv = CrossValidator(
        estimator=rf,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=2,  # Reduced from 3
        parallelism=1,  # Serial execution to save memory
        seed=42
    )
    
    # Train the model
    print("Training Random Forest model...")
    cvModel = cv.fit(train_df)
    best_model = cvModel.bestModel
    
    print("Random Forest training complete.")
    
    return best_model

def build_gradient_boosting_model(train_df):
    """Build and train a Gradient Boosting model with minimal memory usage."""
    print("Building Gradient Boosting model...")
    
    # Create Gradient Boosting model with modest defaults
    gbt = GBTClassifier(
        featuresCol="features", 
        labelCol="target",
        maxIter=20,           # Reduced from 100
        maxDepth=5,           # Reduced from 10
        maxBins=32,           # Reduced from 128
        minInstancesPerNode=5, # Increased from 1
        seed=42
    )
    
    # Define simplified parameter grid
    paramGrid = ParamGridBuilder() \
        .addGrid(gbt.maxIter, [10, 20]) \
        .addGrid(gbt.maxDepth, [3, 5]) \
        .build()
    
    # Create cross-validator
    evaluator = BinaryClassificationEvaluator(
        labelCol="target", 
        metricName="areaUnderROC"
    )
    
    # Create cross-validator with fewer folds
    cv = CrossValidator(
        estimator=gbt,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=2,  # Reduced from 3
        parallelism=1,  # Serial execution to save memory
        seed=42
    )
    
    # Train the model
    print("Training Gradient Boosting model...")
    cvModel = cv.fit(train_df)
    best_model = cvModel.bestModel
    
    print("Gradient Boosting training complete.")
    
    return best_model

def train_models(train_df, test_df):
    """Train multiple models and select the best one."""
    print("Training models...")
    
    # Dictionary to store all trained models
    models = {}
    metrics_list = []
    
    # Train Logistic Regression model
    try:
        print("\n====== Logistic Regression ======")
        lr_model = build_logistic_regression_model(train_df)
        models["logistic_regression"] = lr_model
        lr_metrics, lr_preds = evaluate_model(lr_model, test_df, "logistic_regression")
        metrics_list.append(lr_metrics)
        print("\nLogistic Regression training completed.")
    except Exception as e:
        print(f"Error training Logistic Regression: {e}")
    
    # Train Random Forest model
    try:
        print("\n====== Random Forest ======")
        rf_model = build_random_forest_model(train_df)
        models["random_forest"] = rf_model
        rf_metrics, rf_preds = evaluate_model(rf_model, test_df, "random_forest")
        metrics_list.append(rf_metrics)
        print("\nRandom Forest training completed.")
    except Exception as e:
        print(f"Error training Random Forest: {e}")
    
    # Train Gradient Boosting model
    try:
        print("\n====== Gradient Boosting ======")
        gbt_model = build_gradient_boosting_model(train_df)
        models["gradient_boosting"] = gbt_model
        gbt_metrics, gbt_preds = evaluate_model(gbt_model, test_df, "gradient_boosting")
        metrics_list.append(gbt_metrics)
        print("\nGradient Boosting training completed.")
    except Exception as e:
        print(f"Error training Gradient Boosting: {e}")
    
    # Identify the best model based on its metrics
    if metrics_list:
        best_model_name, best_metrics = save_model_results(models, metrics_list)
        print(f"\nBest model: {best_model_name}")
        print(f"Best model AUC: {best_metrics['auc']:.4f}")
        print(f"Best model Accuracy: {best_metrics['accuracy']:.4f}")
        return models, metrics_list
    else:
        print("No models were successfully trained.")
        return {}, []

def evaluate_model(model, test_df, model_name):
    """Evaluate the model on the test dataset."""
    print(f"Evaluating {model_name} model...")
    
    # Make predictions
    predictions = model.transform(test_df)
    
    # Create evaluators
    binary_evaluator = BinaryClassificationEvaluator(labelCol="target", 
                                                   metricName="areaUnderROC")
    multi_evaluator = MulticlassClassificationEvaluator(labelCol="target", 
                                                      predictionCol="prediction")
    
    # Calculate metrics
    auc = binary_evaluator.evaluate(predictions)
    accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
    precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
    recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})
    f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
    
    # Calculate confusion matrix
    from pyspark.sql.functions import col
    
    # Extract TP, FP, TN, FN
    tp = predictions.filter((col("prediction") == 1.0) & (col("target") == 1.0)).count()
    fp = predictions.filter((col("prediction") == 1.0) & (col("target") == 0.0)).count()
    tn = predictions.filter((col("prediction") == 0.0) & (col("target") == 0.0)).count()
    fn = predictions.filter((col("prediction") == 0.0) & (col("target") == 1.0)).count()
    
    print(f"Confusion Matrix for {model_name}:")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    
    # Compute specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Store metrics in a dictionary
    metrics = {
        "model_name": model_name,
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn
    }
    
    # Print metrics
    print(f"{model_name} Model Metrics:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            print(f"  {metric_name}: {metric_value:.4f}")
    
    return metrics, predictions

def get_feature_importance(model, feature_names, model_name):
    """Extract feature importance from the model if available."""
    feature_importance = {}
    
    try:
        if model_name == "random_forest":
            importance = model.featureImportances.toArray()
            for i, importance_val in enumerate(importance):
                feature_importance[i] = float(importance_val)
        elif model_name == "logistic_regression":
            # For logistic regression, coefficients can be used as importance
            weights = model.coefficients.toArray()
            for i, weight in enumerate(weights):
                feature_importance[i] = abs(float(weight))
        
        print(f"Feature importance for {model_name}:")
        for i, importance_val in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  Feature {i}: {importance_val:.4f}")
        
    except Exception as e:
        print(f"Error getting feature importance: {e}")
    
    return feature_importance

def save_model_results(models, metrics_list, output_dir=MODELS_DIR):
    """Save the trained models and their metrics."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save metrics to a JSON file
    metrics_path = os.path.join(output_dir, "model_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_list, f, indent=4)
    
    print(f"Saved model metrics to {metrics_path}")
    
    # Find best model based on AUC
    best_model_idx = max(range(len(metrics_list)), key=lambda i: metrics_list[i]['auc'])
    best_model_name = metrics_list[best_model_idx]['model_name']
    best_metric = metrics_list[best_model_idx]
    
    # Save the best model
    best_model = models[best_model_name]
    best_model_path = os.path.join(output_dir, f"best_model_{best_model_name}")
    
    try:
        best_model.write().overwrite().save(best_model_path)
        print(f"Saved best model ({best_model_name}) to {best_model_path}")
    except Exception as e:
        print(f"Warning: Could not save model: {e}")
    
    # Save best model parameters to a text file
    params_path = os.path.join(output_dir, f"best_model_params_{best_model_name}.txt")
    with open(params_path, 'w') as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"AUC: {best_metric['auc']:.4f}\n")
        f.write(f"Accuracy: {best_metric['accuracy']:.4f}\n")
        f.write(f"Precision: {best_metric['precision']:.4f}\n")
        f.write(f"Recall: {best_metric['recall']:.4f}\n")
        f.write(f"F1 Score: {best_metric['f1']:.4f}\n")
        if 'specificity' in best_metric:
            f.write(f"Specificity: {best_metric['specificity']:.4f}\n")
    
    print(f"Saved best model parameters to {params_path}")
    
    return best_model_name, best_metric

# ----- VISUALIZATION -----

def plot_class_distribution(df, target_col='target', output_path=VISUALIZATIONS_DIR):
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
    plt.xlabel('Heart Disease Present (Yes) vs Absent (No)')
    plt.ylabel('Count')
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(output_path, "class_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved class distribution plot to {plot_path}")

def plot_correlation_heatmap(df, output_path=VISUALIZATIONS_DIR):
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
    plot_path = os.path.join(output_path, "correlation_heatmap.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved correlation heatmap to {plot_path}")

def plot_feature_distributions(df, output_path=VISUALIZATIONS_DIR):
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
        ax.legend(['No Disease (0.0)', 'Disease (1.0)'])
    
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(output_path, "feature_distributions.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved feature distributions plot to {plot_path}")

def plot_model_comparison(metrics_list, output_path=VISUALIZATIONS_DIR):
    """Plot model comparison based on metrics."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Extract model names and metrics
    model_names = [metrics['model_name'] for metrics in metrics_list]
    accuracy = [metrics['accuracy'] for metrics in metrics_list]
    precision = [metrics['precision'] for metrics in metrics_list]
    recall = [metrics['recall'] for metrics in metrics_list]
    f1 = [metrics['f1'] for metrics in metrics_list]
    auc_values = [metrics['auc'] for metrics in metrics_list]
    specificity = [metrics.get('specificity', 0) for metrics in metrics_list]  # Include new metric
    
    # Create a dataframe for easier plotting
    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc_values,
        'Specificity': specificity
    })
    
    # Melt the dataframe for easier plotting
    melted_df = pd.melt(metrics_df, id_vars=['Model'], 
                      value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Specificity'],
                      var_name='Metric', value_name='Value')
    
    # Plot comparison
    plt.figure(figsize=(16, 10))
    chart = sns.barplot(x='Model', y='Value', hue='Metric', data=melted_df, palette='viridis')
    
    # Add values on top of bars
    for p in chart.patches:
        chart.annotate(f'{p.get_height():.3f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, rotation=90,
                    xytext=(0, 5), textcoords='offset points')
    
    plt.title('Model Performance Comparison', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0, 1.1)
    plt.legend(title='Metric', loc='upper right', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    plot_path = os.path.join(output_path, "model_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved model comparison plot to {plot_path}")

# ----- MAIN PIPELINE -----

def main():
    """Main function to run the entire pipeline."""
    try:
        print("=" * 80)
        print("HEART DISEASE PREDICTION PIPELINE")
        print("=" * 80)
        
        # Create Spark session
        spark = create_spark_session()
        
        # Set log level to reduce output
        spark.sparkContext.setLogLevel("WARN")
        
        # Create directories if they don't exist
        create_directories()
        
        print("\n" + "=" * 30 + " STEP 1: DATA PREPROCESSING " + "=" * 30)
        
        # Load data
        df = load_data(DATASET_PATH)
        
        if df is None:
            print("Error: Could not load data. Exiting.")
            return
            
        # Preprocess data
        try:
            processed_df, preprocessing_model = preprocess_data(df)
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            # Try a simplified preprocessing approach if the original fails
            print("Attempting simplified preprocessing...")
            try:
                # Simple preprocessing without complex transformations
                processed_df = df.withColumnRenamed("Heart Disease Status", "target")
                processed_df = processed_df.withColumn("target", 
                                           when(processed_df["target"] == "Yes", 1.0)
                                           .otherwise(0.0))
                
                # Handle missing numerical values
                for col_name in processed_df.columns:
                    if processed_df.select(col_name).dtypes[0][1] != 'string' and col_name != 'target':
                        processed_df = processed_df.withColumn(
                            col_name, 
                            when(col(col_name).isNull(), lit(0)).otherwise(col(col_name))
                        )
                
                # Convert categorical to numeric
                for col_name in processed_df.columns:
                    if processed_df.select(col_name).dtypes[0][1] == 'string':
                        indexer = StringIndexer(inputCol=col_name, 
                                              outputCol=f"{col_name}_indexed",
                                              handleInvalid="keep")
                        processed_df = indexer.fit(processed_df).transform(processed_df)
                        processed_df = processed_df.drop(col_name).withColumnRenamed(f"{col_name}_indexed", col_name)
                
                # Assemble features
                feature_cols = [col_name for col_name in processed_df.columns if col_name != 'target']
                assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
                processed_df = assembler.transform(processed_df)
                processed_df = processed_df.select("target", "features")
                
                print("Simplified preprocessing completed.")
            except Exception as e2:
                print(f"Simplified preprocessing also failed: {e2}")
                return
        
        # Handle imbalanced data
        try:
            balanced_df = handle_imbalanced_data(processed_df, spark)
        except Exception as e:
            print(f"Error during class balancing: {e}. Using original dataset.")
            balanced_df = processed_df
            
        # Split data into training and testing sets
        train_df, test_df = split_data(balanced_df)
        
        # Save processed data
        try:
            save_processed_data(train_df, test_df)
        except Exception as e:
            print(f"Warning: Could not save processed data: {e}")
        
        print("\n" + "=" * 30 + " STEP 2: MODEL TRAINING AND EVALUATION " + "=" * 30)
        
        # Train models with error handling
        models, metrics_list = train_models(train_df, test_df)
        
        if not models or not metrics_list:
            print("Error: No models were successfully trained. Exiting.")
            return
            
        # Identify and print the best model
        best_model_idx = max(range(len(metrics_list)), key=lambda i: metrics_list[i]['auc'])
        best_model_name = metrics_list[best_model_idx]['model_name']
        best_metrics = metrics_list[best_model_idx]
        
        print("\n" + "=" * 30 + " STEP 3: RESULTS VISUALIZATION " + "=" * 30)
        
        # Visualize results only if we have metrics
        if metrics_list:
            try:
                # Plot model comparison
                plot_model_comparison(metrics_list)
                
                # Plot ROC curves if supported
                if 'logistic_regression' in models:
                    plot_roc_curve(models['logistic_regression'], test_df, "Logistic Regression")
            except Exception as e:
                print(f"Warning: Could not create visualizations: {e}")
        
        print("\n" + "=" * 30 + " PIPELINE COMPLETED SUCCESSFULLY " + "=" * 30)
        print(f"Best model: {best_model_name}")
        print(f"Best model AUC: {best_metrics['auc']:.4f}")
        print(f"Best model Accuracy: {best_metrics['accuracy']:.4f}")
        
        # Stop Spark session
        spark.stop()
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        # Attempt to stop Spark session if it exists
        try:
            if 'spark' in locals():
                spark.stop()
        except:
            pass

if __name__ == "__main__":
    main()
