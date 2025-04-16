import os
import sys
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from pyspark.sql import SparkSession

# Import custom modules
sys.path.append("/Users/drake/Documents/UWE/BDA/Heart-Disease-Prediction---BDA")
from feature_engineering import (
    create_medical_features, encode_categorical_features
)

# Define constants
MODEL_PATH = "/Users/drake/Downloads/output/models/gradient_boosting_model.pkl"
METRICS_PATH = "/Users/drake/Downloads/output/results/model_metrics.json"

# Create Flask app
app = Flask(__name__)

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


@app.route("/features", methods=["GET"])
def get_features():
    """
    Endpoint to get information about expected input features.
    """
    feature_info = {
        'features': [
            {'name': 'age', 'type': 'number', 'min': 20, 'max': 100, 'required': True},
            {'name': 'gender', 'type': 'select', 'options': [1, 2], 'required': True},
            {'name': 'height', 'type': 'number', 'min': 100, 'max': 250, 'required': True},
            {'name': 'weight', 'type': 'number', 'min': 20, 'max': 200, 'required': True},
            {'name': 'ap_hi', 'type': 'number', 'min': 80, 'max': 250, 'required': True},
            {'name': 'ap_lo', 'type': 'number', 'min': 40, 'max': 150, 'required': True},
            {'name': 'cholesterol', 'type': 'select', 'options': [1, 2, 3], 'required': True},
            {'name': 'gluc', 'type': 'select', 'options': [1, 2, 3], 'required': True},
            {'name': 'smoke', 'type': 'select', 'options': [0, 1], 'required': True},
            {'name': 'alco', 'type': 'select', 'options': [0, 1], 'required': True},
            {'name': 'active', 'type': 'select', 'options': [0, 1], 'required': True}
        ]
    }
    return jsonify(feature_info)

@app.route("/metrics", methods=["GET"])
def get_metrics():
    """
    Endpoint to get model metrics.
    """
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                metrics = json.load(f)
            return jsonify(metrics)
        else:
            return jsonify({'error': 'Metrics not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint for making predictions on new data.
    """
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        processed_data = preprocess_input(input_df)
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0, 1]
        result = {
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_level": get_risk_level(probability)
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def preprocess_input(input_df):
    """
    Preprocess input data for prediction.
    
    Parameters:
    -----------
    input_df : pandas.DataFrame
        Input data
        
    Returns:
    --------
    numpy.ndarray
        Preprocessed data ready for prediction
    """
    # Create a copy to avoid modifying the original
    df = input_df.copy()
    
    # Calculate age in years if not provided
    if 'age_years' not in df.columns and 'age' in df.columns:
        df['age_years'] = df['age']
    
    # Calculate BMI if not provided
    if 'bmi' not in df.columns and 'height' in df.columns and 'weight' in df.columns:
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    
    # Create blood pressure category if not provided
    if 'bp_category' not in df.columns and 'ap_hi' in df.columns and 'ap_lo' in df.columns:
        df['bp_category'] = df.apply(lambda row: categorize_blood_pressure(row['ap_hi'], row['ap_lo']), axis=1)
    
    # Create medical features
    df = create_medical_features(df)
    
    # Encode categorical features
    df = encode_categorical_features(df)
    
    # Select relevant features (same as used during training)
    selected_features = [
        'age_years', 'bmi', 'ap_hi', 'ap_lo', 'pulse_pressure', 'mean_arterial_pressure',
        'gender_1', 'gender_2', 'cholesterol_1', 'cholesterol_2', 'cholesterol_3',
        'gluc_1', 'gluc_2', 'gluc_3', 'smoke_0', 'smoke_1', 'alco_0', 'alco_1',
        'active_0', 'active_1', 'bp_category_Normal', 'bp_category_Elevated',
        'bp_category_Hypertension Stage 1', 'bp_category_Hypertension Stage 2',
        'risk_factors_count', 'hypertensive_inactive'
    ]
    
    # Keep only features that exist in the DataFrame
    available_features = [f for f in selected_features if f in df.columns]
    
    # Fill missing features with 0
    for feature in selected_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Return numpy array with selected features
    return df[selected_features].values

def categorize_blood_pressure(systolic, diastolic):
    """
    Categorize blood pressure according to AHA guidelines.
    
    Parameters:
    -----------
    systolic : int
        Systolic blood pressure (mmHg)
    diastolic : int
        Diastolic blood pressure (mmHg)
        
    Returns:
    --------
    str
        Blood pressure category
    """
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif (120 <= systolic < 130) and diastolic < 80:
        return "Elevated"
    elif (130 <= systolic < 140) or (80 <= diastolic < 90):
        return "Hypertension Stage 1"
    elif systolic >= 140 or diastolic >= 90:
        return "Hypertension Stage 2"
    else:
        return "Normal"  # Default case

def get_risk_level(probability):
    """
    Get risk level based on prediction probability.
    
    Parameters:
    -----------
    probability : float
        Prediction probability
        
    Returns:
    --------
    str
        Risk level (Low, Moderate, High, Very High)
    """
    if probability < 0.25:
        return "Low"
    elif probability < 0.5:
        return "Moderate"
    elif probability < 0.75:
        return "High"
    else:
        return "Very High"

def get_recommendation(data, prediction, probability):
    """
    Get personalized recommendations based on input data and prediction.
    
    Parameters:
    -----------
    data : dict
        Input data
    prediction : int
        Prediction (0: no disease, 1: disease)
    probability : float
        Prediction probability
        
    Returns:
    --------
    dict
        Personalized recommendations
    """
    recommendations = []
    
    # Basic recommendation for everyone
    recommendations.append("Regular check-ups with healthcare provider")
    
    # Blood pressure recommendations
    if data.get('ap_hi', 0) >= 130 or data.get('ap_lo', 0) >= 80:
        recommendations.append("Monitor blood pressure regularly")
        recommendations.append("Consider consulting with a healthcare provider about blood pressure management")
    
    # Lifestyle recommendations
    if data.get('smoke', 0) == 1:
        recommendations.append("Consider smoking cessation programs")
    
    if data.get('alco', 0) == 1:
        recommendations.append("Limit alcohol consumption")
    
    if data.get('active', 0) == 0:
        recommendations.append("Increase physical activity (aim for at least 150 minutes of moderate exercise per week)")
    
    # Weight recommendations
    bmi = data.get('weight', 0) / ((data.get('height', 0) / 100) ** 2)
    if bmi >= 25:
        recommendations.append("Consider weight management strategies")
    
    # Cholesterol and glucose recommendations
    if data.get('cholesterol', 0) > 1:
        recommendations.append("Monitor cholesterol levels and consider dietary changes")
    
    if data.get('gluc', 0) > 1:
        recommendations.append("Monitor blood glucose levels and consider dietary changes")
    
    # High risk recommendations
    if prediction == 1 and probability >= 0.5:
        recommendations.append("Urgent consultation with a cardiologist is recommended")
        recommendations.append("Consider comprehensive cardiovascular assessment")
    
    return recommendations

@app.route("/health", methods=["GET"])
def health_check():
    """
    Endpoint for health check.
    """
    return jsonify({"status": "healthy"})

@app.route("/", methods=["GET"])
def home():
    """
    Home endpoint with API documentation.
    """
    return jsonify({
        "name": "Cardiovascular Disease Prediction API",
        "version": "1.0.0",
        "description": "API for predicting cardiovascular disease risk",
        "endpoints": {
            "/predict": {
                "method": "POST",
                "description": "Make a prediction",
                "parameters": {
                    "age": "Age in years",
                    "gender": "Gender (1: female, 2: male)",
                    "height": "Height in cm",
                    "weight": "Weight in kg",
                    "ap_hi": "Systolic blood pressure",
                    "ap_lo": "Diastolic blood pressure",
                    "cholesterol": "Cholesterol level (1: normal, 2: above normal, 3: well above normal)",
                    "gluc": "Glucose level (1: normal, 2: above normal, 3: well above normal)",
                    "smoke": "Smoking status (0: non-smoker, 1: smoker)",
                    "alco": "Alcohol intake (0: doesn't drink, 1: drinks)",
                    "active": "Physical activity (0: inactive, 1: active)"
                }
            },
            "/health": {
                "method": "GET",
                "description": "Health check endpoint"
            }
        }
    })

def start_server(host="0.0.0.0", port=5001, debug=False):
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start the Cardiovascular Disease Prediction API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=5001, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    print(f"Starting server on {args.host}:{args.port}")
    start_server(host=args.host, port=args.port, debug=args.debug)