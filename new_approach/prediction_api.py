#!/usr/bin/env python3
"""
Cardiovascular Disease Prediction API

This script provides a Flask API for making predictions using the trained models.
It loads the best model and provides endpoints for single and batch predictions.
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import PipelineModel
import xgboost as xgb

app = Flask(__name__)

# Constants
MODELS_DIR = "/Users/drake/Documents/UWE/BDA/Heart-Disease-Prediction---BDA/output/models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_model.pkl")  # Assuming XGBoost is the best model

# Load the best model
with open(BEST_MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Feature names (must match the order used during training)
FEATURE_NAMES = [
    "age_years", "gender", "height", "weight", "ap_hi", "ap_lo", 
    "cholesterol", "gluc", "smoke", "alco", "active", "bmi"
]

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions.
    
    Expected JSON format:
    {
        "age_years": 50,
        "gender": 1,  # 1: Female, 2: Male
        "height": 165,
        "weight": 70,
        "ap_hi": 120,
        "ap_lo": 80,
        "cholesterol": 1,  # 1: Normal, 2: Above Normal, 3: Well Above Normal
        "gluc": 1,  # 1: Normal, 2: Above Normal, 3: Well Above Normal
        "smoke": 0,  # 0: No, 1: Yes
        "alco": 0,   # 0: No, 1: Yes
        "active": 1  # 0: No, 1: Yes
    }
    
    Returns:
    {
        "prediction": 0 or 1,  # 0: No cardiovascular disease, 1: Cardiovascular disease
        "probability": 0.XX,   # Probability of cardiovascular disease
        "risk_factors": [      # Top risk factors contributing to the prediction
            {"feature": "feature_name", "importance": X.XX},
            ...
        ]
    }
    """
    try:
        # Get data from request
        data = request.json
        
        # Calculate BMI if not provided
        if "bmi" not in data:
            height_m = data["height"] / 100  # Convert height from cm to m
            data["bmi"] = data["weight"] / (height_m ** 2)
        
        # Create feature vector
        features = [data.get(feature, 0) for feature in FEATURE_NAMES]
        features_array = np.array([features])
        
        # Make prediction
        prediction = int(model.predict(features_array)[0])
        probability = float(model.predict_proba(features_array)[0, 1])
        
        # Get feature importances for this prediction
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_array)
        
        # Get top risk factors
        feature_importance = list(zip(FEATURE_NAMES, shap_values[0]))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        risk_factors = [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in feature_importance[:5]  # Top 5 risk factors
        ]
        
        return jsonify({
            "prediction": prediction,
            "probability": probability,
            "risk_factors": risk_factors
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    Endpoint for making batch predictions.
    
    Expected JSON format:
    {
        "data": [
            {
                "age_years": 50,
                "gender": 1,
                ...
            },
            ...
        ]
    }
    
    Returns:
    {
        "predictions": [
            {
                "id": 0,
                "prediction": 0 or 1,
                "probability": 0.XX
            },
            ...
        ]
    }
    """
    try:
        # Get data from request
        data = request.json["data"]
        
        # Process each record
        results = []
        for i, record in enumerate(data):
            # Calculate BMI if not provided
            if "bmi" not in record:
                height_m = record["height"] / 100  # Convert height from cm to m
                record["bmi"] = record["weight"] / (height_m ** 2)
            
            # Create feature vector
            features = [record.get(feature, 0) for feature in FEATURE_NAMES]
            features_array = np.array([features])
            
            # Make prediction
            prediction = int(model.predict(features_array)[0])
            probability = float(model.predict_proba(features_array)[0, 1])
            
            results.append({
                "id": i,
                "prediction": prediction,
                "probability": probability
            })
        
        return jsonify({"predictions": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)