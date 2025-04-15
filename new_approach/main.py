#!/usr/bin/env python3
"""
Main Script for Cardiovascular Disease Prediction

This script orchestrates the entire workflow for cardiovascular disease prediction:
1. Data loading and preprocessing
2. Feature engineering
3. Model training and evaluation
4. Results visualization and reporting
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Import custom modules
from utils.data_preprocessing import (
    load_data, clean_data, explore_data, split_data,
    load_spark_data, clean_spark_data, split_spark_data
)
from feature_engineering import (
    create_medical_features, encode_categorical_features,
    select_features, create_feature_pipeline
)
from model_training import (
    train_sklearn_model, train_spark_model,
    save_model, load_model
)
from evaluation import (
    evaluate_sklearn_model, evaluate_spark_model,
    plot_confusion_matrix, plot_roc_curve
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cardiovascular Disease Prediction')
    
    parser.add_argument('--data_path', type=str, 
                        default='Dataset/cardio_data_processed.csv',
                        help='Path to the dataset')

    parser.add_argument('--output_dir', type=str, 
                        default='output',
                        help='Directory to save outputs')
    
    parser.add_argument('--use_spark', action='store_true',
                        help='Use Spark for processing')
    
    parser.add_argument('--explore', action='store_true',
                        help='Perform exploratory data analysis')
    
    parser.add_argument('--model', type=str, 
                        choices=['rf', 'gb', 'lr', 'svm', 'all'],
                        default='rf',
                        help='Model to train (rf: Random Forest, gb: Gradient Boosting, '
                             'lr: Logistic Regression, svm: Support Vector Machine, all: All models)')
    
    parser.add_argument('--feature_selection', type=str,
                        choices=['f_classif', 'mutual_info', 'rfe', 'none'],
                        default='none',
                        help='Feature selection method')
    
    parser.add_argument('--n_features', type=int, 
                        default=10,
                        help='Number of features to select')
    
    parser.add_argument('--save_model', action='store_true',
                        help='Save the trained model')
    
    return parser.parse_args()

def setup_directories(output_dir):
    """Create necessary directories for outputs."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)

def get_model(model_name):
    """Get the model based on the model name."""
    if model_name == 'rf':
        return RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    elif model_name == 'gb':
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    elif model_name == 'lr':
        return LogisticRegression(
            C=1.0,
            penalty='l2',
            solver='liblinear',
            random_state=42
        )
    elif model_name == 'svm':
        return SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def run_sklearn_workflow(args):
    """Run the scikit-learn workflow."""
    print("\n=== Running scikit-learn workflow ===\n")
    
    # Load and preprocess data
    df = load_data(args.data_path)
    df_clean = clean_data(df)
    
    # Exploratory data analysis
    if args.explore:
        explore_data(df_clean)
    
    # Feature engineering
    df_features = create_medical_features(df_clean)
    df_encoded = encode_categorical_features(df_features)
    
    # Feature selection
    if args.feature_selection != 'none':
        selected_features, df_selected = select_features(
            df_encoded, 
            method=args.feature_selection, 
            k=args.n_features
        )
    else:
        df_selected = df_encoded
    
    # Split data
    train_df, val_df, test_df = split_data(df_selected)
    
    # Prepare features and target
    X_train = train_df.drop('cardio', axis=1)
    y_train = train_df['cardio']
    X_val = val_df.drop('cardio', axis=1)
    y_val = val_df['cardio']
    X_test = test_df.drop('cardio', axis=1)
    y_test = test_df['cardio']
    
    # Remove ID column if present
    if 'id' in X_train.columns:
        X_train = X_train.drop('id', axis=1)
        X_val = X_val.drop('id', axis=1)
        X_test = X_test.drop('id', axis=1)
    
    # Train and evaluate models
    results = {}
    
    if args.model == 'all':
        models = ['rf', 'gb', 'lr', 'svm']
    else:
        models = [args.model]
    
    for model_name in models:
        print(f"\n--- Training {model_name} model ---\n")
        
        # Get model
        model = get_model(model_name)
        
        # Train model
        trained_model = train_sklearn_model(model, X_train, y_train, X_val, y_val)
        
        # Evaluate model
        eval_results = evaluate_sklearn_model(trained_model, X_test, y_test)
        results[model_name] = eval_results
        
        # Print results
        print(f"\nResults for {model_name}:")
        print(f"Accuracy: {eval_results['Accuracy']:.4f}")
        print(f"Precision: {eval_results['Precision']:.4f}")
        print(f"Recall: {eval_results['Recall']:.4f}")
        print(f"F1 Score: {eval_results['F1 Score']:.4f}")
        print(f"AUC: {eval_results['AUC']:.4f}")
        print("\nClassification Report:")
        print(eval_results['Classification Report'])
        
        # Plot confusion matrix
        plot_confusion_matrix(
            eval_results['y_true'], 
            eval_results['y_pred'],
            output_path=os.path.join(args.output_dir, 'visualizations', f'{model_name}_confusion_matrix.png')
        )
        
        # Plot ROC curve
        plot_roc_curve(
            eval_results['y_true'], 
            eval_results['y_pred_proba'],
            output_path=os.path.join(args.output_dir, 'visualizations', f'{model_name}_roc_curve.png')
        )
        
        # Save model
        if args.save_model:
            save_model(
                trained_model, 
                os.path.join(args.output_dir, 'models', f'{model_name}_model.pkl')
            )
    
    # Compare models
    if len(models) > 1:
        print("\n=== Model Comparison ===\n")
        
        # Create comparison table
        comparison = pd.DataFrame({
            model_name: {
                'Accuracy': results[model_name]['Accuracy'],
                'Precision': results[model_name]['Precision'],
                'Recall': results[model_name]['Recall'],
                'F1 Score': results[model_name]['F1 Score'],
                'AUC': results[model_name]['AUC']
            } for model_name in models
        }).T
        
        print(comparison)
        
        # Save comparison
        comparison.to_csv(os.path.join(args.output_dir, 'results', 'model_comparison.csv'))
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        comparison.plot(kind='bar', figsize=(12, 8))
        plt.title('Model Comparison')
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.xticks(rotation=0)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'visualizations', 'model_comparison.png'))
    
    return results

def run_spark_workflow(args):
    """Run the Spark workflow."""
    print("\n=== Running Spark workflow ===\n")
    
    # Create Spark session
    spark = SparkSession.builder \
    .appName("CardiovascularDiseasePrediction") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.hadoop.security.authentication", "simple") \
    .getOrCreate()
    
    try:
        # Load and preprocess data
        spark_df = load_spark_data(spark, args.data_path)
        spark_df_clean = clean_spark_data(spark_df)
        
        # Split data
        train_df, val_df, test_df = split_spark_data(spark_df_clean)
        
        # Train model
        # Note: For simplicity, we're using a basic Random Forest model in Spark
        from pyspark.ml.classification import RandomForestClassifier as SparkRandomForestClassifier
        
        # Define feature columns
        categorical_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bp_category']
        numeric_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
        
        # Create feature pipeline
        from feature_engineering import create_spark_feature_pipeline
        pipeline = create_spark_feature_pipeline(categorical_cols, numeric_cols)
        
        # Add Random Forest to the pipeline
        from pyspark.ml import Pipeline
        rf = SparkRandomForestClassifier(
            labelCol="cardio",
            featuresCol="scaled_features",
            numTrees=100,
            maxDepth=10,
            seed=42
        )
        
        pipeline = Pipeline(stages=pipeline.getStages() + [rf])
        
        # Train model
        model = pipeline.fit(train_df)
        
        # Evaluate model
        eval_results = evaluate_spark_model(model, test_df)
        
        # Print results
        print("\nResults for Spark Random Forest:")
        print(f"Accuracy: {eval_results['Accuracy']:.4f}")
        print(f"Precision: {eval_results['Precision']:.4f}")
        print(f"Recall: {eval_results['Recall']:.4f}")
        print(f"F1 Score: {eval_results['F1 Score']:.4f}")
        print(f"AUC: {eval_results['AUC']:.4f}")
        
        # Save model
        if args.save_model:
            model.write().overwrite().save(os.path.join(args.output_dir, 'models', 'spark_rf_model'))
        
        return eval_results
    
    finally:
        # Stop Spark session
        spark.stop()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup directories
    setup_directories(args.output_dir)
    
    # Run workflow
    if args.use_spark:
        results = run_spark_workflow(args)
    else:
        results = run_sklearn_workflow(args)
    
    print("\n=== Workflow completed successfully ===\n")

if __name__ == "__main__":
    main()