#!/usr/bin/env python3
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import pandas as pd
import numpy as np
import joblib
import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

app = Flask(__name__, static_folder='../webapp/build')
CORS(app)

# Create Spark session
def create_spark_session(app_name="HeartDiseaseAPI"):
    """Create and return a Spark session."""
    # Set Java home environment variable if not already set
    if "JAVA_HOME" not in os.environ:
        # Use the WSL Java path
        os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
        print(f"Set JAVA_HOME to {os.environ['JAVA_HOME']}")
    
    # Configure Spark to use the correct Python executable
    os.environ["PYSPARK_PYTHON"] = sys.executable
    
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.ui.port", "4050") \
        .config("spark.local.dir", "/tmp/spark-temp") \
        .master("local[*]") \
        .getOrCreate()

# Global variables
spark = None
model = None
model_type = None
feature_columns = None

def load_model():
    """Load the trained model."""
    global model, model_type, feature_columns
    
    # Get the model metrics to find the best model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_file = os.path.join(base_dir, 'models/model_metrics.json')
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics_list = json.load(f)
        
        # Find the best model based on AUC
        best_metric = max(metrics_list, key=lambda x: x['auc'])
        model_type = best_metric['model_name']
        
        # Load the model using joblib
        model_path = os.path.join(base_dir, f'models/best_model_{model_type}.joblib')
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                print(f"Loaded {model_type} model from {model_path}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
    
    return False

# Load feature columns from the processed dataset
def load_feature_columns():
    """Load feature columns from the processed dataset."""
    global feature_columns
    
    try:
        # Load a sample of the data to get the feature names
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, 'data/cardio_data_processed.csv')
        df = pd.read_csv(data_path)
        
        # Get all columns except the target and non-feature columns
        feature_columns = [col for col in df.columns if col not in ['id', 'cardio', 'bp_category']]
        print(f"Loaded {len(feature_columns)} feature columns")
        
        return True
    except Exception as e:
        print(f"Error loading feature columns: {e}")
        return False

# Preprocess input data
def preprocess_input(input_data):
    """Preprocess the input data for prediction."""
    global spark, feature_columns, model
    
    try:
        # Initialize Spark if not already done
        if spark is None:
            spark = create_spark_session()
            
        # Convert input to pandas DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Print debug information
        print(f"Input data: {input_data}")
        print(f"Feature columns: {feature_columns}")
        print(f"Feature columns length: {len(feature_columns)}")
        
        # Get the expected feature count from the model
        # For XGBoost models, we can check the feature count directly
        if hasattr(model, 'n_features_'):
            expected_feature_count = model.n_features_
        elif hasattr(model, 'n_features_in_'):
            expected_feature_count = model.n_features_in_
        else:
            # Default to the length of feature_columns
            expected_feature_count = len(feature_columns)
        
        print(f"Model expects {expected_feature_count} features")
        
        # Handle bp_category_encoded specially
        if 'bp_category_encoded' in feature_columns and 'bp_category' in input_data:
            # Map blood pressure category to encoded value
            bp_category = input_data['bp_category']
            if bp_category == "Normal":
                input_df['bp_category_encoded'] = 0
            elif bp_category == "Elevated":
                input_df['bp_category_encoded'] = 1
            elif bp_category == "Hypertension Stage 1":
                input_df['bp_category_encoded'] = 2
            elif bp_category == "Hypertension Stage 2":
                input_df['bp_category_encoded'] = 3
            else:
                input_df['bp_category_encoded'] = 0
        
        # Ensure we have all the required features
        missing_features = [col for col in feature_columns if col not in input_df.columns]
        if missing_features:
            print(f"Missing features: {missing_features}")
            for col in missing_features:
                input_df[col] = 0  # Default value for missing features
        
        # Remove any extra columns not in feature_columns
        extra_columns = [col for col in input_df.columns if col not in feature_columns and col not in ['id', 'cardio', 'bp_category']]
        if extra_columns:
            print(f"Extra columns being removed: {extra_columns}")
            input_df = input_df.drop(columns=extra_columns)
        
        # If we need to adjust the feature count to match the model's expectation
        if expected_feature_count != len(feature_columns):
            print(f"Adjusting feature count from {len(feature_columns)} to {expected_feature_count}")
            # If we have too many features, remove the last ones
            if len(feature_columns) > expected_feature_count:
                adjusted_features = feature_columns[:expected_feature_count]
                input_df = input_df[adjusted_features]
            # If we have too few features, we can't proceed
            else:
                raise ValueError(f"Model expects {expected_feature_count} features, but only {len(feature_columns)} are available")
        else:
            # Select only the feature columns in the correct order
            input_df = input_df[feature_columns]
        
        print(f"Final feature count: {input_df.shape[1]}")
        
        # Convert to numpy array for prediction
        features = input_df.values.astype(float)
        
        return features
    except Exception as e:
        print(f"Error preprocessing input: {e}")
        return None

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        input_data = request.json
        
        # Validate required fields
        required_fields = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                          'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        
        for field in required_fields:
            if field not in input_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Calculate derived features
        input_data['age_years'] = int(input_data['age'] // 365)
        input_data['bmi'] = input_data['weight'] / ((input_data['height'] / 100) ** 2)
        
        # Determine blood pressure category
        systolic = input_data['ap_hi']
        diastolic = input_data['ap_lo']
        
        if systolic < 120 and diastolic < 80:
            bp_category = "Normal"
        elif (120 <= systolic < 130) and diastolic < 80:
            bp_category = "Elevated"
        elif (130 <= systolic < 140) or (80 <= diastolic < 90):
            bp_category = "Hypertension Stage 1"
        elif systolic >= 140 or diastolic >= 90:
            bp_category = "Hypertension Stage 2"
        else:
            bp_category = "Normal"
        
        input_data['bp_category'] = bp_category
        
        # Preprocess input for model
        features = preprocess_input(input_data)
        
        if features is None:
            return jsonify({'error': 'Failed to preprocess input data'}), 500
        
        # Make prediction
        try:
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
            
            return jsonify({
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': 'High' if prediction == 1 else 'Low',
                'model_used': model_type
            })
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return jsonify({'error': f'Error making prediction: {str(e)}'}), 500
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_file = os.path.join(base_dir, 'models/model_metrics.json')
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            model_metrics = json.load(f)
        return jsonify(model_metrics)
    else:
        return jsonify({'error': 'Model metrics not found'}), 404

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
