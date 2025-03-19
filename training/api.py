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
from flask import Flask, request, jsonify
import joblib
import json

app = Flask(__name__)

# Load models and metrics
models = {
    'logistic_regression': joblib.load('models/logistic_regression.pkl'),
    'random_forest': joblib.load('models/random_forest.pkl'),
    'xgboost': joblib.load('models/xgboost.pkl')
}

with open('models/model_metrics.json', 'r') as f:
    model_metrics = json.load(f)

@app.route('/api/predict', methods=['POST'])
def predict():
    input_data = request.json
    model_name = input_data.get('model_name', 'logistic_regression')
    model = models.get(model_name)
    
    if not model:
        return jsonify({'error': 'Model not found'}), 404
    
    prediction = model.predict([input_data['features']])
    return jsonify({'prediction': prediction[0]})

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify(model_metrics)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

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
