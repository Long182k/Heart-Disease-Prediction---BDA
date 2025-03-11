#!/usr/bin/env python3
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel, GBTClassificationModel
from pyspark.ml.feature import VectorAssembler
import joblib

app = Flask(__name__, static_folder='../webapp/build')
CORS(app)

# Create Spark session
def create_spark_session(app_name="HeartDiseaseAPI"):
    """Create and return a Spark session."""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.ui.port", "4050") \
        .config("spark.local.dir", "/tmp/spark-temp") \
        .getOrCreate()

# Global variables
spark = create_spark_session()
model = None
model_type = None
feature_columns = None

def load_model():
    """Load the trained model."""
    global model, model_type, feature_columns
    
    # Get the model metrics to find the best model
    metrics_file = '../models/model_metrics.json'
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics_list = json.load(f)
        
        # Find the best model based on AUC
        best_metric = max(metrics_list, key=lambda x: x['auc'])
        model_type = best_metric['model_name']
        
        # Load the model
        model_path = f'../models/best_model_{model_type}'
        if os.path.exists(model_path):
            try:
                if model_type == 'logistic_regression':
                    model = LogisticRegressionModel.load(model_path)
                elif model_type == 'random_forest':
                    model = RandomForestClassificationModel.load(model_path)
                elif model_type == 'gradient_boosting':
                    model = GBTClassificationModel.load(model_path)
                
                print(f"Loaded {model_type} model from {model_path}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
    
    return False

# Load feature columns from the original dataset
def load_feature_columns():
    """Load feature columns from the original dataset."""
    global feature_columns
    
    try:
        # Load a sample of the data to get the feature names
        data_path = '../Dataset/archive/heart_disease.csv'
        df = pd.read_csv(data_path)
        
        # Get all columns except the target
        feature_columns = [col for col in df.columns if col != 'target']
        print(f"Loaded {len(feature_columns)} feature columns")
        
        return True
    except Exception as e:
        print(f"Error loading feature columns: {e}")
        return False

# Preprocess input data
def preprocess_input(input_data):
    """Preprocess the input data for prediction."""
    try:
        # Convert input to pandas DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Convert pandas DataFrame to Spark DataFrame
        spark_df = spark.createDataFrame(input_df)
        
        # Ensure we have all the required features
        for col in feature_columns:
            if col not in input_data:
                raise ValueError(f"Missing feature: {col}")
        
        # Create feature vector
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        processed_df = assembler.transform(spark_df)
        
        return processed_df
    except Exception as e:
        print(f"Error preprocessing input: {e}")
        return None

# Routes
@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a prediction for heart disease."""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please initialize the model first.'}), 500
    
    try:
        # Get input data from request
        input_data = request.json
        
        # Preprocess input data
        processed_df = preprocess_input(input_data)
        if processed_df is None:
            return jsonify({'error': 'Error preprocessing input data'}), 400
        
        # Make prediction
        prediction = model.transform(processed_df)
        
        # Extract prediction result
        result = prediction.select('prediction', 'probability').collect()[0]
        
        # Convert NumPy data types to Python native types for JSON serialization
        prediction_value = float(result['prediction'])
        probability = result['probability'].toArray().tolist()
        
        response = {
            'prediction': prediction_value,
            'probability': probability,
            'predicted_class': 'Heart Disease Present' if prediction_value == 1 else 'No Heart Disease',
            'model_type': model_type
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get model metrics."""
    metrics_file = '../models/model_metrics.json'
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics_list = json.load(f)
        return jsonify(metrics_list)
    else:
        return jsonify({'error': 'Metrics file not found'}), 404

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get feature columns."""
    if feature_columns is None:
        return jsonify({'error': 'Feature columns not loaded'}), 500
    
    return jsonify({'features': feature_columns})

# Serve the React app
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # Load the model and feature columns
    if load_model() and load_feature_columns():
        print("API server is ready")
    else:
        print("Warning: Model or feature columns could not be loaded, some endpoints may not work")
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)
