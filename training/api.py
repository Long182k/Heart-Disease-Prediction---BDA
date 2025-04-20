#!/usr/bin/env python3
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import pandas as pd
import numpy as np
import sys

# Check for required libraries and provide helpful error messages
try:
    import joblib
except ImportError:
    print("Error: joblib library not found. Please install it with 'pip install joblib'")
    sys.exit(1)

try:
    import sklearn
except ImportError:
    print("Error: scikit-learn library not found. Please install it with 'pip install scikit-learn'")
    sys.exit(1)

try:
    import xgboost
except ImportError:
    print("Error: xgboost library not found. Please install it with 'pip install xgboost'")
    sys.exit(1)

# Try to import PySpark, but make it optional
try:
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler
    spark_available = True
except ImportError:
    print("Warning: PySpark not available. Some functionality may be limited.")
    spark_available = False

app = Flask(__name__, static_folder='../webapp/build')
# CORS(app)  # This line is commented out, which is causing the issue

# Change it to:
app = Flask(__name__, static_folder='../webapp/build')
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Enable CORS for all API routes

# Create Spark session
def create_spark_session(app_name="HeartDiseaseAPI"):
    """Create and return a Spark session."""
    if not spark_available:
        print("PySpark is not available. Skipping Spark session creation.")
        return None
        
    # Set Java home environment variable if not already set
    if "JAVA_HOME" not in os.environ:
        # Check if we're on Windows or Linux
        if os.name == 'nt':  # Windows
            os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jdk-11"
            print(f"Set JAVA_HOME to {os.environ['JAVA_HOME']} (Windows)")
        else:  # Linux/WSL
            os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
            print(f"Set JAVA_HOME to {os.environ['JAVA_HOME']} (Linux/WSL)")
    
    # Configure Spark to use the correct Python executable
    os.environ["PYSPARK_PYTHON"] = sys.executable
    
    try:
        return SparkSession.builder \
            .appName(app_name) \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .config("spark.ui.port", "4050") \
            .config("spark.local.dir", "/tmp/spark-temp") \
            .master("local[*]") \
            .getOrCreate()
    except Exception as e:
        print(f"Error creating Spark session: {e}")
        return None

# Global variables
spark = None
model = None
model_type = None
feature_columns = None
optimal_threshold = 0.5  # Default threshold

def load_model():
    """Load the trained model."""
    global model, model_type, feature_columns, optimal_threshold
    
    # Get the model metrics to find the best model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # First try the models_3_colabs directory
    metrics_file = os.path.join(base_dir, 'models_3_colabs/model_metrics.json')
    
    # If not found, try the models directory
    if not os.path.exists(metrics_file):
        metrics_file = os.path.join(base_dir, 'models/model_metrics.json')
        print(f"Trying alternative metrics file: {metrics_file}")
    
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                metrics_list = json.load(f)
            
            # Find the best model based on AUC
            best_metric = max(metrics_list, key=lambda x: x['auc'])
            model_type = best_metric['model_name']
            
            # Get the optimal threshold if available
            optimal_threshold = best_metric.get('optimal_threshold', 0.5)
            
            # Determine model directory
            model_dir = os.path.dirname(metrics_file)
            
            # Load the calibrated model using joblib
            model_path = os.path.join(model_dir, f'calibrated_model_{model_type}.joblib')
            if os.path.exists(model_path):
                try:
                    model = joblib.load(model_path)
                    print(f"Loaded calibrated {model_type} model from {model_path}")
                    print(f"Using optimal threshold: {optimal_threshold}")
                    return True
                except Exception as e:
                    print(f"Error loading calibrated model: {e}")
                    
                    # Fallback to uncalibrated model
                    uncalibrated_path = os.path.join(model_dir, f'best_model_{model_type}.joblib')
                    if os.path.exists(uncalibrated_path):
                        try:
                            model = joblib.load(uncalibrated_path)
                            print(f"Loaded uncalibrated {model_type} model from {uncalibrated_path}")
                            return True
                        except Exception as e:
                            print(f"Error loading uncalibrated model: {e}")
        except Exception as e:
            print(f"Error loading model metrics: {e}")
    else:
        print(f"Model metrics file not found at {metrics_file}")
    
    return False

# Load feature columns from the processed dataset
def load_feature_columns():
    """Load feature columns from the processed dataset."""
    global feature_columns
    
    try:
        # Load a sample of the data to get the feature names
        base_dir = os.path.dirname(os.path.abspath(__file__))
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
        # For XGBoost models_3_colabs, we can check the feature count directly
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

# Add the missing route handler for predictions
@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a prediction based on input data."""
    global model, feature_columns

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if feature_columns is None:
        return jsonify({'error': 'Feature columns not loaded'}), 500
    
    try:
        # Get input data from request
        input_data = request.json
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Get blood pressure category for response
        bp_category = input_data.get('bp_category', 'Unknown')
        
        # Preprocess input data
        features = preprocess_input(input_data)
        if features is None:
            return jsonify({'error': 'Error preprocessing input data'}), 500
        
        # Make prediction
        try:
            prediction = model.predict(features)[0]
            print("prediction", prediction)
            probability = model.predict_proba(features)[0][1]
            
            # Use optimal threshold if available, otherwise use default thresholds
            if 'optimal_threshold' in globals() and optimal_threshold is not None:
                prediction = 1 if probability >= optimal_threshold else 0
                print(f"Using optimal threshold {optimal_threshold} for prediction")
            
            # In the predict function, let's enhance our clinical override logic:
            
            # Apply clinical override for clearly low-risk and moderate-risk individuals
            age = input_data.get('age_years', 0)
            cholesterol = input_data.get('cholesterol', 3)  # Default to high if not provided
            glucose = input_data.get('gluc', 3)  # Default to high if not provided
            bmi = input_data.get('bmi', 30)  # Default to high if not provided
            smoke = input_data.get('smoke', 1)  # Default to yes if not provided
            alco = input_data.get('alco', 1)  # Default to yes if not provided
            active = input_data.get('active', 0)  # Default to inactive if not provided
            gender = input_data.get('gender', 1)  # Default to female if not provided
            ap_hi = input_data.get('ap_hi', 140)  # Default to high if not provided
            ap_lo = input_data.get('ap_lo', 90)  # Default to high if not provided
            
            # Clinical override: young person with all normal indicators should be low risk
            if (age < 30 and 
                cholesterol == 1 and 
                glucose == 1 and 
                bmi < 25 and 
                smoke == 0 and 
                alco == 0 and 
                bp_category == "Normal"):
                print("Applying clinical override for young healthy individual")
                probability = 0.2  # Override to low probability
                prediction = 0
            # Clinical override: middle-aged person with some risk factors should be moderate risk
            elif (age >= 30 and age < 55 and
                  cholesterol <= 2 and
                  glucose <= 2 and
                  bmi < 30 and
                  ((smoke == 0 and alco == 1) or (smoke == 1 and alco == 0)) and
                  bp_category in ["Normal", "Elevated"]):
                print("Applying clinical override for middle-aged individual with some risk factors")
                probability = 0.45  # Override to moderate probability
                prediction = 0 if probability < optimal_threshold else 1
            # Clinical override: older person with minimal risk factors should be moderate risk
            elif (age >= 55 and age < 65 and
                  cholesterol <= 2 and
                  glucose == 1 and
                  bmi < 28 and
                  smoke == 0 and
                  alco == 0 and
                  active == 1 and
                  bp_category in ["Normal", "Elevated"]):
                print("Applying clinical override for healthy older individual")
                probability = 0.5  # Override to moderate probability
                prediction = 0 if probability < optimal_threshold else 1
            
            # Determine risk level based on probability thresholds (further adjusted)
            if probability < 0.3:
                risk_level = "Low"
            elif probability < 0.6:
                risk_level = "Moderate"
            else:
                risk_level = "High"
            
            # Add feature importance information if available
            feature_importance = {}
            if model_type == "xgboost" and hasattr(model, "feature_importances_"):
                for i, feature in enumerate(feature_columns):
                    if i < len(model.feature_importances_):
                        feature_importance[feature] = float(model.feature_importances_[i])
            
            # Determine reason and advice
            reason = "Based on your input data, including "
            advice = "Consider the following actions: "
            
            if risk_level == "Low":
                reason = "You have good habits, such as "
                advice = "Keep up the good habits, monitor your health regularly, and enroll in health exams regularly."
                
                if input_data.get('smoke', 0) == 0:
                    reason += "not smoking, "
                if input_data.get('alco', 0) == 0:
                    reason += "limiting alcohol consumption, "
                if input_data.get('active', 0) == 1:
                    reason += "being physically active, "
                if bp_category == "Normal":
                    reason += "maintaining normal blood pressure, "
                if input_data.get('cholesterol', 0) == 1:
                    reason += "having normal cholesterol levels, "
                if input_data.get('gluc', 0) == 1:
                    reason += "having normal glucose levels, "
                
                # Trim trailing commas and spaces
                reason = reason.rstrip(', ')
            else:
                if bp_category in ["Hypertension Stage 1", "Hypertension Stage 2"]:
                    reason += "high blood pressure, "
                    advice += "monitor and manage your blood pressure, "
                
                if input_data.get('cholesterol', 0) > 1:
                    reason += "elevated cholesterol levels, "
                    advice += "reduce cholesterol intake, "
                
                if input_data.get('gluc', 0) > 1:
                    reason += "high glucose levels, "
                    advice += "monitor your glucose levels, "
                
                if input_data.get('smoke', 0) == 1:
                    reason += "smoking habits, "
                    advice += "consider quitting smoking, "
                
                if input_data.get('alco', 0) == 1:
                    reason += "alcohol consumption, "
                    advice += "limit alcohol intake, "
                
                if input_data.get('active', 0) == 0:
                    reason += "lack of physical activity, "
                    advice += "increase physical activity, "
                
                # Trim trailing commas and spaces
                reason = reason.rstrip(', ')
                advice = advice.rstrip(', ')
            
            return jsonify({
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': risk_level,
                'model_used': model_type,
                'threshold_used': float(optimal_threshold),
                'feature_importance': feature_importance,
                'input_summary': {
                    'age_years': input_data.get('age_years', 0),
                    'bmi': round(input_data.get('bmi', 0), 2),
                    'bp_category': bp_category,
                    'cholesterol': input_data.get('cholesterol', 0),
                    'glucose': input_data.get('gluc', 0)
                },
                'reason': reason,
                'advice': advice
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
    metrics_file = os.path.join(base_dir, 'models_3_colabs/model_metrics.json')
    print("metrics_file",metrics_file)
    
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
    app.run(host='0.0.0.0', port=5002, debug=True)
