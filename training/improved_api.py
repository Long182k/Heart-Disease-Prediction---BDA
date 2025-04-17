from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import pandas as pd
import numpy as np
import sys
import joblib
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='../webapp/build')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Global variables
model = None
model_type = None
feature_columns = None
optimal_threshold = 0.5
scaler = None

def load_model():
    """Load the trained model."""
    global model, model_type, feature_columns, optimal_threshold, scaler
    
    # Get the model metrics to find the best model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # First try the models_improved directory
    metrics_file = os.path.join(base_dir, 'models_improved/model_metrics.json')
    print(f"Trying metrics file: {metrics_file}")
    
    # If not found, try the models_3_colabs directory
    if not os.path.exists(metrics_file):
        metrics_file = os.path.join(base_dir, 'models_3_colabs/model_metrics.json')
        print(f"Trying alternative metrics file: {metrics_file}")
    
    # If still not found, try the models directory
    if not os.path.exists(metrics_file):
        metrics_file = os.path.join(base_dir, 'models/model_metrics.json')
        print(f"Trying alternative metrics file: {metrics_file}")
    
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                metrics_list = json.load(f)
            

            print(f"Loaded metrics for {len(metrics_list)} models")
            
            # Find the best model based on AUC
            best_metric = max(metrics_list, key=lambda x: x['auc'])
            model_type = best_metric['model_name']
            print(f"Best model based on AUC: {model_type} (AUC: {best_metric['auc']})")
            
            # Get the optimal threshold if available
            optimal_threshold = best_metric.get('optimal_threshold', 0.5)
            
            # Determine model directory
            model_dir = os.path.dirname(metrics_file)
            print(f"Model directory: {model_dir}")
            
            # Try loading models in this order: calibrated model, best model, ensemble model
            model_paths = [
                os.path.join(model_dir, f'calibrated_model_{model_type}.joblib'),
                os.path.join(model_dir, f'best_model_{model_type}.joblib'),
                os.path.join(model_dir, 'ensemble_model.joblib')
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        print(f"Attempting to load model from: {model_path}")
                        model = joblib.load(model_path)
                        print(f"Successfully loaded model from {model_path}")
                        print(f"Using optimal threshold: {optimal_threshold}")
                        
                        # Try to load the scaler
                        scaler_path = os.path.join(model_dir, 'scaler.joblib')
                        if os.path.exists(scaler_path):
                            try:
                                scaler = joblib.load(scaler_path)
                                print(f"Loaded scaler from {scaler_path}")
                            except Exception as e:
                                print(f"Warning: Could not load scaler: {e}")
                        else:
                            print(f"No scaler found at {scaler_path}")
                        
                        return True
                    except Exception as e:
                        print(f"Error loading model from {model_path}: {e}")
            
            # If we get here, none of the model paths worked
            print("Failed to load any model. Trying fallback approach...")
            
            # Fallback: try to load any model in the directory
            for model_name in ["xgboost", "random_forest", "gradient_boosting", "logistic_regression", "ensemble"]:
                for prefix in ["calibrated_model_", "best_model_"]:
                    model_path = os.path.join(model_dir, f"{prefix}{model_name}.joblib")
                    if os.path.exists(model_path):
                        try:
                            model = joblib.load(model_path)
                            model_type = model_name
                            print(f"Loaded fallback model: {model_path}")
                            return True
                        except Exception as e:
                            print(f"Error loading fallback model {model_path}: {e}")
            
            print("All model loading attempts failed")
            return False
        except Exception as e:
            print(f"Error loading model metrics: {e}")
    else:
        print(f"Model metrics file not found at {metrics_file}")
    
    return False

def load_feature_columns():
    """Load feature columns for the model."""
    global feature_columns
    
    try:
        # First try to load feature names from the improved models directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        feature_names_path = os.path.join(base_dir, 'models_improved/feature_names.json')
        
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                feature_columns = json.load(f)
            print(f"Loaded {len(feature_columns)} feature columns from feature_names.json")
            return True
        
        # If not found, try to load from the processed dataset
        data_path = os.path.join(base_dir, 'data/cardio_data_processed.csv')
        
        if not os.path.exists(data_path):
            print(f"Data file not found at {data_path}")
            # Try to find the data file in other locations
            parent_dir = os.path.dirname(base_dir)
            alternative_paths = [
                os.path.join(parent_dir, 'data/cardio_data_processed.csv'),
                os.path.join(base_dir, 'cardio_data_processed.csv')
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    data_path = alt_path
                    print(f"Found data file at alternative location: {data_path}")
                    break
            else:
                print("Could not find data file in any expected location")
                return False
        
        df = pd.read_csv(data_path)
        
        # Get all columns except the target and non-feature columns
        feature_columns = [col for col in df.columns if col not in ['id', 'cardio', 'bp_category']]
        print(f"Loaded {len(feature_columns)} feature columns from dataset")
        
        return True
    except Exception as e:
        print(f"Error loading feature columns: {e}")
        return False

def preprocess_input(input_data):
    """Preprocess the input data for prediction."""
    global feature_columns, model, scaler
    
    try:
        # Convert input to pandas DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Print debug information
        print(f"Input data: {input_data}")
        print(f"Feature columns: {feature_columns}")
        print(f"Feature columns length: {len(feature_columns)}")
        
        # Get the expected feature count from the model
        if hasattr(model, 'n_features_in_'):
            expected_feature_count = model.n_features_in_
            print(f"Model expects {expected_feature_count} features")
            
            # Print the actual feature names if available
            if hasattr(model, 'feature_names_in_'):
                print(f"Model was trained with these features: {model.feature_names_in_}")
        else:
            # Default to the length of feature_columns
            expected_feature_count = len(feature_columns)
            print(f"Model expects {expected_feature_count} features (based on feature_columns)")
        
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
                
        # Add bp_category_numeric if it's in the feature columns
        if 'bp_category_numeric' in feature_columns and 'bp_category' in input_data:
            # Map blood pressure category to numeric value using the same mapping
            bp_category = input_data['bp_category']
            bp_category_mapping = {
                'Normal': 0,
                'Elevated': 1,
                'Hypertension Stage 1': 2,
                'Hypertension Stage 2': 3,
                'Hypertensive Crisis': 4
            }
            input_df['bp_category_numeric'] = bp_category_mapping.get(bp_category, 0)
            print(f"Mapped bp_category '{bp_category}' to bp_category_numeric: {input_df['bp_category_numeric'].values[0]}")
        
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
        
        # Select only the feature columns in the correct order
        input_df = input_df[feature_columns]
        
        print(f"Final feature count: {input_df.shape[1]}")
        
        # Apply scaling if scaler is available
        if scaler is not None:
            features = scaler.transform(input_df.values)
            print("Applied scaling to input features")
        else:
            # Convert to numpy array for prediction
            features = input_df.values.astype(float)
        
        return features
    except Exception as e:
        print(f"Error preprocessing input: {e}")
        return None

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a prediction based on input data."""
    global model, feature_columns, optimal_threshold
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if feature_columns is None:
        return jsonify({'error': 'Feature columns not loaded'}), 500
    
    try:
        # Get input data from request
        input_data = request.json
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # # Calculate BMI if not provided but weight and height are available
        # if 'weight' in input_data and 'height' in input_data and ('bmi' not in input_data or input_data['bmi'] == 0):
        #     weight_kg = float(input_data['weight'])
        #     height_m = float(input_data['height']) / 100  # Convert height from cm to m
        #     calculated_bmi = weight_kg / (height_m * height_m)
        #     input_data['bmi'] = round(calculated_bmi, 1)  # Round to 1 decimal place
        #     print(f"Calculated BMI: {input_data['bmi']} from weight: {weight_kg}kg and height: {height_m}m")
        
        # Get blood pressure category for response
        bp_category = input_data.get('bp_category', 'Unknown')
        
        # Preprocess input data
        features = preprocess_input(input_data)
        if features is None:
            return jsonify({'error': 'Error preprocessing input data'}), 500
        
        # Make prediction
        try:
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]
            
            # Use optimal threshold if available, otherwise use default thresholds
            if optimal_threshold is not None:
                prediction = 1 if probability >= optimal_threshold else 0
                print(f"Using optimal threshold {optimal_threshold} for prediction")
            
            # Extract all relevant features for clinical assessment
            age = input_data.get('age_years', 0)
            cholesterol = input_data.get('cholesterol', 3)  # Default to high if not provided
            glucose = input_data.get('gluc', 3)  # Default to high if not provided
            bmi = input_data.get('bmi', 30)  # Default to high if not provided
            smoke = input_data.get('smoke', 1)  # Default to yes if not provided
            alco = input_data.get('alco', 1)  # Default to yes if not provided
            active = input_data.get('active', 0)  # Default to inactive if not provided
            ap_hi = input_data.get('ap_hi', 140)  # Default to high if not provided
            ap_lo = input_data.get('ap_lo', 90)  # Default to high if not provided
            
            # Initialize risk level
            risk_level = None
            risk_factors_count = 0
            
            # Count risk factors
            if cholesterol > 1:
                risk_factors_count += 1
            if glucose > 1:
                risk_factors_count += 1
            if smoke == 1:
                risk_factors_count += 1
            if bmi >= 25:
                risk_factors_count += 1
            if alco == 1:
                risk_factors_count += 1
            if active == 0:
                risk_factors_count += 1
            if bp_category != "Normal":
                risk_factors_count += 1
            
            # HIGH RISK CONDITIONS
            if (
                # Older adults with risk factors
                (age >= 65 and risk_factors_count >= 1) or
                # Middle-aged with multiple risk factors
                (age >= 45 and age < 65 and risk_factors_count >= 3) or
                # Hypertension Stage 2
                (ap_hi >= 140 or ap_lo >= 90) or
                # Hypertensive Crisis
                (ap_hi > 180 or ap_lo > 120) or
                # Well above normal cholesterol with other risks
                (cholesterol == 3 and risk_factors_count >= 2) or
                # Diabetes with other risks
                (glucose == 3 and risk_factors_count >= 2) or
                # Smoker with multiple risks
                (smoke == 1 and risk_factors_count >= 3) or
                # Severe obesity with risks
                (bmi >= 35 and risk_factors_count >= 2)
            ):
                risk_level = "High"
                probability = max(probability, 0.7)  # Ensure probability reflects high risk
                prediction = 1
            
            # MODERATE RISK CONDITIONS (if not already high risk)
            elif (
                # Middle-aged with some risk factors
                (age >= 45 and age < 65 and risk_factors_count >= 1) or
                # Younger adults with multiple risk factors
                (age >= 20 and age < 45 and risk_factors_count >= 3) or
                # Hypertension Stage 1
                (ap_hi >= 130 and ap_hi < 140) or (ap_lo >= 80 and ap_lo < 90) or
                # Elevated BP with other risks
                (ap_hi >= 120 and ap_hi < 130 and ap_lo < 80 and risk_factors_count >= 1) or
                # Above normal cholesterol
                (cholesterol == 2) or
                # Prediabetes
                (glucose == 2) or
                # Overweight with risks
                (bmi >= 25 and bmi < 30 and risk_factors_count >= 1) or
                # Obesity Class 1
                (bmi >= 30 and bmi < 35) or
                # Regular alcohol consumption
                (alco == 1) or
                # Physical inactivity with other risks
                (active == 0 and risk_factors_count >= 1)
            ):
                risk_level = "Moderate"
                probability = max(min(probability, 0.69), 0.3)  # Ensure probability reflects moderate risk
                prediction = 1 if probability >= optimal_threshold else 0
            
            # LOW RISK CONDITIONS (if not already high or moderate risk)
            else:
                risk_level = "Low"
                probability = min(probability, 0.29)  # Ensure probability reflects low risk
                prediction = 0
            
            print(f"Clinical assessment: {risk_level} risk level with {risk_factors_count} risk factors")
            
            # Add feature importance information if available
            feature_importance = {}
            if model_type in ["xgboost", "random_forest", "gradient_boosting"] and hasattr(model, "feature_importances_"):
                for i, feature in enumerate(feature_columns):
                    if i < len(model.feature_importances_):
                        feature_importance[feature] = float(model.feature_importances_[i])
            
            # Determine reason and advice
            reason = "Based on your input data, including "
            advice = "Consider the following actions: "
            
            if risk_level == "Low":
                reason += f"your age ({age}), blood pressure ({bp_category}), and other factors, you have a low risk of heart disease."
                advice += "Continue maintaining a healthy lifestyle with regular exercise and a balanced diet."
            elif risk_level == "Moderate":
                reason += f"your age ({age}), blood pressure ({bp_category}), and other health indicators, you have a moderate risk of heart disease."
                advice += "Schedule a check-up with your doctor, consider lifestyle modifications, and monitor your blood pressure regularly."
            else:  # High risk
                reason += f"your age ({age}), blood pressure ({bp_category}), cholesterol level, and other risk factors, you have a high risk of heart disease."
                advice += "Consult with a healthcare professional as soon as possible, follow a heart-healthy diet, exercise regularly, and consider medication if prescribed."
            
            # Create response
            response = {
                'prediction': int(prediction),  # 0: Absence, 1: Presence of cardiovascular disease
                'probability': float(probability),
                'cardio': int(prediction),  # Adding explicit cardio field to match target variable
                'risk_level': risk_level,  # Keeping risk_level for interpretability
                'model_type': model_type,
                'bp_category': bp_category,
                'reason': reason,
                'advice': advice,
                'feature_importance': feature_importance
            }
            
            return jsonify(response)
        except Exception as e:
            print(f"Error making prediction: {e}")
            return jsonify({'error': f'Error making prediction: {e}'}), 500
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': f'Error processing request: {e}'}), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get feature columns."""
    if feature_columns is None:
        return jsonify({'error': 'Feature columns not loaded'}), 500
    
    return jsonify({'features': feature_columns})

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get model metrics."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # First try the improved models directory
    metrics_file = os.path.join(base_dir, 'models_improved/model_metrics.json')
    
    # If not found, try the models_3_colabs directory
    if not os.path.exists(metrics_file):
        metrics_file = os.path.join(base_dir, 'models_3_colabs/model_metrics.json')
    
    # If still not found, try the models directory
    if not os.path.exists(metrics_file):
        metrics_file = os.path.join(base_dir, 'models/model_metrics.json')
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            model_metrics = json.load(f)
        return jsonify(model_metrics)
    else:
        return jsonify({'error': 'Model metrics file not found'}), 404

@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Get model information
    model_info = {
        'model_type': model_type,
        'optimal_threshold': optimal_threshold,
        'feature_count': len(feature_columns) if feature_columns else 0,
        'features': feature_columns,
        'has_scaler': scaler is not None
    }
    
    # Add feature importance if available
    if model_type in ["xgboost", "random_forest", "gradient_boosting"] and hasattr(model, "feature_importances_"):
        feature_importance = {}
        for i, feature in enumerate(feature_columns):
            if i < len(model.feature_importances_):
                feature_importance[feature] = float(model.feature_importances_[i])
        model_info['feature_importance'] = feature_importance
    
    return jsonify(model_info)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Serve the static files from the React app."""
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

def inspect_model_features():
    """Inspect and print the features that the model was trained on."""
    global model, feature_columns
    
    print("\n=== MODEL FEATURE INSPECTION ===")
    print(f"Model type: {model_type}")
    
    if hasattr(model, 'feature_names_in_'):
        print(f"Model was trained with these {len(model.feature_names_in_)} features:")
        for i, feature in enumerate(model.feature_names_in_):
            print(f"  {i+1}. {feature}")
    
    print(f"\nCurrent feature_columns ({len(feature_columns)}):")
    for i, feature in enumerate(feature_columns):
        print(f"  {i+1}. {feature}")
    
    # Check for mismatches
    if hasattr(model, 'feature_names_in_'):
        missing_in_model = [f for f in feature_columns if f not in model.feature_names_in_]
        missing_in_columns = [f for f in model.feature_names_in_ if f not in feature_columns]
        
        if missing_in_model:
            print(f"\nWARNING: These features are in feature_columns but not in the model:")
            for f in missing_in_model:
                print(f"  - {f}")
        
        if missing_in_columns:
            print(f"\nWARNING: These features are in the model but not in feature_columns:")
            for f in missing_in_columns:
                print(f"  - {f}")
    
    print("=== END OF INSPECTION ===\n")

if __name__ == '__main__':
    # Load the model and feature columns
    if load_model() and load_feature_columns():
        print("API server is ready")
        # Inspect the model features
        inspect_model_features()
    else:
        print("Warning: Model or feature columns could not be loaded, some endpoints may not work")
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=True)