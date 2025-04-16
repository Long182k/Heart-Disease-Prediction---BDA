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
MODEL_PATH = "/Users/drake/Documents/UWE/BDA/Heart-Disease-Prediction---BDA/gradient_boosting_model_test_13.pkl"
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
            {'name': 'active', 'type': 'select', 'options': [0, 1], 'required': True},
            {'name': 'bmi', 'type': 'number', 'min': 10, 'max': 60, 'required': True},
            {'name': 'bp_category', 'type': 'select', 'options': [
                "Normal", "Elevated", "Hypertension Stage 1", "Hypertension Stage 2"
            ], 'required': True}
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
    Endpoint for making predictions.
    """
    data = request.get_json()
    required_features = [
        'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
        'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'bp_category'
    ]
    
    # Check for missing features
    missing = [f for f in required_features if f not in data]
    if missing:
        # Try to calculate missing features if possible
        if 'bmi' in missing and 'height' in data and 'weight' in data:
            data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)
            missing.remove('bmi')
        
        if 'bp_category' in missing and 'ap_hi' in data and 'ap_lo' in data:
            data['bp_category'] = categorize_blood_pressure(data['ap_hi'], data['ap_lo'])
            missing.remove('bp_category')
        
        # If still missing required features, return error
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

    input_df = pd.DataFrame([data])
    # Preprocess input to match training features
    processed_input = preprocess_input(input_df)

        # After loading your model
    print(f"Number of features expected by the model: {model.n_features_in_}")
    # For models that store feature names
    if hasattr(model, 'feature_names_in_'):
        print(f"Feature names: {model.feature_names_in_}")
        print(f"Number of features: {len(model.feature_names_in_)}")

    # Predict
    pred = model.predict(processed_input)[0]
    prob = model.predict_proba(processed_input)[0][1] if hasattr(model, "predict_proba") else None
    
    # Get risk level and recommendations
    risk_level = get_risk_level(prob) if prob is not None else None
    recommendations = get_recommendation(data, pred, prob) if prob is not None else []

    return jsonify({
        "prediction": int(pred),
        "probability": float(prob) if prob is not None else None,
        "risk_level": risk_level,
        "recommendations": recommendations
    })

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
    
    # Define the required features
    required_features = [
        'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
        'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'bp_category'
    ]
    
    # Calculate age in years if not provided
    if 'age_years' not in df.columns and 'age' in df.columns:
        df['age_years'] = df['age']
    
    # Calculate BMI if not provided
    if 'bmi' not in df.columns and 'height' in df.columns and 'weight' in df.columns:
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    
    # Create blood pressure category if not provided
    if 'bp_category' not in df.columns and 'ap_hi' in df.columns and 'ap_lo' in df.columns:
        df['bp_category'] = df.apply(lambda row: categorize_blood_pressure(row['ap_hi'], row['ap_lo']), axis=1)
    
    # Check for missing required features
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Fill missing features with default values
        for feature in missing_features:
            if feature in ['gender', 'cholesterol', 'gluc']:
                df[feature] = 1  # Default to normal/female
            elif feature in ['smoke', 'alco', 'active']:
                df[feature] = 0  # Default to no
            else:
                df[feature] = 0  # Default to 0 for numeric features
    
    # Ensure categorical features are properly encoded if needed
    if df['bp_category'].dtype == 'object':
        # If the model expects encoded features, encode them
        if hasattr(model, 'feature_names_in_') and any('bp_category_' in feat for feat in model.feature_names_in_):
            df = encode_categorical_features(df)
    
    # Return the features in the correct order
    return df[required_features].values

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