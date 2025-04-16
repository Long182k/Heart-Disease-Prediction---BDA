#!/usr/bin/env python3
"""
Flask backend server for Heart Disease Prediction App
Serves the trained model for the React frontend
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__, static_folder='webapp/build', static_url_path='/')
CORS(app)

# Project directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

# Load the best model
def load_model():
    try:
        model = joblib.load(BEST_MODEL_PATH)
        print(f"Model loaded successfully from {BEST_MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

best_model = load_model()

# Define the features expected by the model
expected_features = [
    'Age', 'Sex', 'Chest Pain Type', 'Resting Blood Pressure', 'Cholesterol', 
    'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate', 'Exercise Angina',
    'ST Depression', 'ST Slope', 'Number of Major Vessels', 'Thalassemia Type',
    'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Resting Heart Rate',
    # Engineered features will be calculated in the backend
]

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        print("Received prediction request with data:", data)
        
        # Prepare input data
        input_df = pd.DataFrame([data])
        
        # Feature engineering - add the same engineered features used during training
        input_df['Age_Cholesterol_Ratio'] = input_df['Age'] / (input_df['Cholesterol'] + 1e-5)
        input_df['HR_BP_Product'] = input_df['Resting Heart Rate'] * input_df['Systolic Blood Pressure']
        input_df['Pulse_Pressure'] = input_df['Systolic Blood Pressure'] - input_df['Diastolic Blood Pressure']
        
        # Handle categorical features
        for col in input_df.select_dtypes(['object']).columns:
            input_df[col] = input_df[col].astype('category').cat.codes
        
        # Ensure all features are numeric
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        
        # Replace any NaN values with median
        input_df.fillna(0, inplace=True)
        
        # Make prediction
        if best_model is not None:
            probability = best_model.predict_proba(input_df)[0][1]
            prediction = 1 if probability >= 0.5 else 0
            
            return jsonify({
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': get_risk_level(probability)
            })
        else:
            return jsonify({
                'error': 'Model not loaded'
            }), 500
    
    except Exception as e:
        print(f"Error making prediction: {e}")
        return jsonify({
            'error': str(e)
        }), 500

def get_risk_level(probability):
    if probability < 0.2:
        return "Low Risk"
    elif probability < 0.5:
        return "Moderate Risk"
    elif probability < 0.7:
        return "High Risk"
    else:
        return "Very High Risk"

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                metrics = json.load(f)
            return jsonify(metrics)
        else:
            return jsonify({'error': 'Metrics not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    feature_info = {
        'features': [
            {'name': 'Age', 'type': 'number', 'min': 20, 'max': 100, 'required': True},
            {'name': 'Sex', 'type': 'select', 'options': ['Male', 'Female'], 'required': True},
            {'name': 'Chest Pain Type', 'type': 'select', 
             'options': ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'], 
             'required': True},
            {'name': 'Resting Blood Pressure', 'type': 'number', 'min': 80, 'max': 200, 'required': True},
            {'name': 'Cholesterol', 'type': 'number', 'min': 100, 'max': 600, 'required': True},
            {'name': 'Fasting Blood Sugar', 'type': 'select', 
             'options': ['Less than 120 mg/dl', 'Greater than 120 mg/dl'], 
             'required': True},
            {'name': 'Resting ECG', 'type': 'select', 
             'options': ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'], 
             'required': True},
            {'name': 'Max Heart Rate', 'type': 'number', 'min': 60, 'max': 220, 'required': True},
            {'name': 'Exercise Angina', 'type': 'select', 'options': ['Yes', 'No'], 'required': True},
            {'name': 'ST Depression', 'type': 'number', 'min': 0, 'max': 10, 'step': 0.1, 'required': True},
            {'name': 'ST Slope', 'type': 'select', 'options': ['Upsloping', 'Flat', 'Downsloping'], 'required': True},
            {'name': 'Number of Major Vessels', 'type': 'number', 'min': 0, 'max': 4, 'required': True},
            {'name': 'Thalassemia Type', 'type': 'select', 
             'options': ['Normal', 'Fixed Defect', 'Reversible Defect'], 
             'required': True},
            {'name': 'Systolic Blood Pressure', 'type': 'number', 'min': 90, 'max': 220, 'required': True},
            {'name': 'Diastolic Blood Pressure', 'type': 'number', 'min': 40, 'max': 130, 'required': True},
            {'name': 'Resting Heart Rate', 'type': 'number', 'min': 40, 'max': 120, 'required': True},
        ]
    }
    return jsonify(feature_info)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
