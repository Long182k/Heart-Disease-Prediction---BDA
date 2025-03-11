#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import joblib
import os
import sys
import json

def create_spark_session(app_name="HeartDiseaseModelTrainer"):
    """Create and return a Spark session."""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.ui.port", "4050") \
        .config("spark.local.dir", "/tmp/spark-temp") \
        .getOrCreate()

def load_processed_data(spark, train_path, test_path):
    """Load the processed training and testing data."""
    print("Loading processed data...")
    train_df = spark.read.csv(train_path, header=True, inferSchema=True)
    test_df = spark.read.csv(test_path, header=True, inferSchema=True)
    
    # Convert back to the right format if needed
    # This may be needed if the data was saved to CSV which loses the vector format
    
    return train_df, test_df

def build_logistic_regression_model(train_df):
    """Build and train a Logistic Regression model with cross-validation."""
    print("Training Logistic Regression model...")
    
    # Initialize the model
    lr = LogisticRegression(featuresCol="features", labelCol="target", 
                            maxIter=10, regParam=0.3, elasticNetParam=0.8)
    
    # Create parameter grid for cross-validation
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1, 0.3]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 0.8]) \
        .addGrid(lr.maxIter, [10, 20, 50]) \
        .build()
    
    # Create a binary classification evaluator
    evaluator = BinaryClassificationEvaluator(labelCol="target", 
                                             metricName="areaUnderROC")
    
    # Create cross-validator
    cv = CrossValidator(estimator=lr,
                       estimatorParamMaps=paramGrid,
                       evaluator=evaluator,
                       numFolds=5,
                       seed=42)
    
    # Train the model using cross-validation
    cv_model = cv.fit(train_df)
    
    # Get the best model
    best_model = cv_model.bestModel
    print(f"Best Logistic Regression model parameters:")
    print(f"  regParam: {best_model.getRegParam()}")
    print(f"  elasticNetParam: {best_model.getElasticNetParam()}")
    print(f"  maxIter: {best_model.getMaxIter()}")
    
    return best_model

def build_random_forest_model(train_df):
    """Build and train a Random Forest model with cross-validation."""
    print("Training Random Forest model...")
    
    # Initialize the model
    rf = RandomForestClassifier(featuresCol="features", labelCol="target", seed=42)
    
    # Create parameter grid for cross-validation
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20, 50]) \
        .addGrid(rf.maxDepth, [5, 10, 15]) \
        .addGrid(rf.minInstancesPerNode, [1, 2, 4]) \
        .build()
    
    # Create a binary classification evaluator
    evaluator = BinaryClassificationEvaluator(labelCol="target",
                                             metricName="areaUnderROC")
    
    # Create cross-validator
    cv = CrossValidator(estimator=rf,
                       estimatorParamMaps=paramGrid,
                       evaluator=evaluator,
                       numFolds=5,
                       seed=42)
    
    # Train the model using cross-validation
    cv_model = cv.fit(train_df)
    
    # Get the best model
    best_model = cv_model.bestModel
    print(f"Best Random Forest model parameters:")
    print(f"  numTrees: {best_model.getNumTrees()}")
    print(f"  maxDepth: {best_model.getMaxDepth()}")
    
    return best_model

def build_gradient_boosting_model(train_df):
    """Build and train a Gradient Boosting Tree model with cross-validation."""
    print("Training Gradient Boosting model...")
    
    # Initialize the model
    gbt = GBTClassifier(featuresCol="features", labelCol="target", maxIter=10, seed=42)
    
    # Create parameter grid for cross-validation
    paramGrid = ParamGridBuilder() \
        .addGrid(gbt.maxIter, [10, 20, 50]) \
        .addGrid(gbt.maxDepth, [5, 10, 15]) \
        .addGrid(gbt.stepSize, [0.1, 0.2, 0.3]) \
        .build()
    
    # Create a binary classification evaluator
    evaluator = BinaryClassificationEvaluator(labelCol="target",
                                             metricName="areaUnderROC")
    
    # Create cross-validator
    cv = CrossValidator(estimator=gbt,
                       estimatorParamMaps=paramGrid,
                       evaluator=evaluator,
                       numFolds=3,  # Using 3 folds for GBT as it's more computational intensive
                       seed=42)
    
    # Train the model using cross-validation
    cv_model = cv.fit(train_df)
    
    # Get the best model
    best_model = cv_model.bestModel
    print(f"Best Gradient Boosting model parameters:")
    print(f"  maxIter: {best_model.getMaxIter()}")
    print(f"  maxDepth: {best_model.getMaxDepth()}")
    print(f"  stepSize: {best_model.getStepSize()}")
    
    return best_model

def evaluate_model(model, test_df, model_name):
    """Evaluate the model on the test dataset."""
    print(f"Evaluating {model_name} model...")
    
    # Make predictions
    predictions = model.transform(test_df)
    
    # Binary classification evaluator for AUC
    binary_evaluator = BinaryClassificationEvaluator(labelCol="target",
                                                   metricName="areaUnderROC")
    auc = binary_evaluator.evaluate(predictions)
    
    # Multiclass evaluator for accuracy, precision, recall, and F1
    multi_evaluator = MulticlassClassificationEvaluator(labelCol="target",
                                                      predictionCol="prediction")
    
    accuracy = multi_evaluator.setMetricName("accuracy").evaluate(predictions)
    precision = multi_evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall = multi_evaluator.setMetricName("weightedRecall").evaluate(predictions)
    f1 = multi_evaluator.setMetricName("f1").evaluate(predictions)
    
    # Create a dictionary with all metrics
    metrics = {
        "model_name": model_name,
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    # Print metrics
    print(f"Metrics for {model_name}:")
    for metric_name, metric_value in metrics.items():
        if metric_name != "model_name":
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

def save_model_results(models, metrics_list, output_dir="../models"):
    """Save the trained models and their metrics."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save metrics to JSON
    metrics_file = os.path.join(output_dir, "model_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics_list, f, indent=4)
    
    print(f"Saved model metrics to {metrics_file}")
    
    # Save the best model based on AUC
    best_metric = max(metrics_list, key=lambda x: x['auc'])
    best_model_name = best_metric['model_name']
    
    best_model = models[best_model_name]
    best_model_path = os.path.join(output_dir, f"best_model_{best_model_name}")
    
    # Save the model using Spark's native save method
    try:
        best_model.save(best_model_path)
        print(f"Saved best model ({best_model_name}) to {best_model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    return best_model_name, best_metric

if __name__ == "__main__":
    # Create Spark session
    spark = create_spark_session()
    
    # Set paths to processed data
    train_path = "../processed_data/train_data.csv"
    test_path = "../processed_data/test_data.csv"
    
    try:
        # Load processed data
        train_df, test_df = load_processed_data(spark, train_path, test_path)
        
        # Build and train models
        lr_model = build_logistic_regression_model(train_df)
        rf_model = build_random_forest_model(train_df)
        gbt_model = build_gradient_boosting_model(train_df)
        
        # Store models in a dictionary
        models = {
            "logistic_regression": lr_model,
            "random_forest": rf_model,
            "gradient_boosting": gbt_model
        }
        
        # Evaluate models
        metrics_list = []
        for model_name, model in models.items():
            metrics, predictions = evaluate_model(model, test_df, model_name)
            metrics_list.append(metrics)
            
            # Get feature importance if available
            get_feature_importance(model, [], model_name)
        
        # Save model results
        best_model_name, best_metric = save_model_results(models, metrics_list)
        
        print(f"Model training completed successfully!")
        print(f"Best model: {best_model_name} with AUC: {best_metric['auc']:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        # Stop Spark session
        spark.stop()
