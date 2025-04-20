#!/usr/bin/env python3
"""
Improved Heart Disease Prediction Model Training
This script implements advanced techniques to improve model performance.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    print("Warning: imblearn.over_sampling.SMOTE could not be imported.")
    print("Will use alternative class balancing method.")
    HAS_SMOTE = False

import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Create output directories
os.makedirs('models_improved', exist_ok=True)
os.makedirs('plots', exist_ok=True)

def load_and_preprocess_data(data_path):
    """Load and preprocess the dataset."""
    print(f"Loading data from {data_path}")
    
    # Try to find the data file
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        # Try to find the data file in other locations
        base_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(base_dir)
        
        # Add more potential Kaggle paths
        alternative_paths = [
            os.path.join(parent_dir, 'data/cardio_data_processed.csv'),
            os.path.join(base_dir, 'cardio_data_processed.csv'),
            '/kaggle/input/cardiovascular-disease/cardio_data_processed.csv',
            '/kaggle/input/cardiovascular-disease/cardio_train.csv',  # Try original dataset name
            '/kaggle/input/cardiovascular-disease-dataset/cardio_data_processed.csv',
            '/kaggle/input/cardiovascular-disease-dataset/cardio_train.csv'
        ]
        
        # Search for any CSV file in the Kaggle input directory
        kaggle_input_dir = '/kaggle/input'
        if os.path.exists(kaggle_input_dir):
            print("Searching for CSV files in Kaggle input directories...")
            for root, dirs, files in os.walk(kaggle_input_dir):
                for file in files:
                    if file.endswith('.csv'):
                        csv_path = os.path.join(root, file)
                        print(f"Found CSV file: {csv_path}")
                        alternative_paths.append(csv_path)
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                data_path = alt_path
                print(f"Found data file at alternative location: {data_path}")
                break
        else:
            raise FileNotFoundError("Could not find data file in any expected location. Please check the dataset path.")
    
    try:
        # First try to use Spark-based preprocessing from data_preprocessing.py
        print("Attempting to use Spark-based distributed preprocessing...")
        from data_preprocessing import preprocess_data
        X_train, X_test, y_train, y_test = preprocess_data(data_path)
        
        # Combine train and test for feature selection and further processing
        X_combined = np.vstack((X_train, X_test))
        y_combined = np.concatenate((y_train, y_test))
        
        print(f"Successfully processed data with Spark: {X_combined.shape[1]} features")
        
        # Load the original data to get feature names
        df = pd.read_csv(data_path)
        
        # Define the expected features
        features = [
            "age", "gender", "height", "weight", "ap_hi", "ap_lo", 
            "cholesterol", "gluc", "smoke", "alco", "active", 
            "age_years", "bmi"
        ]
        
        # Add bp_category_numeric if available
        if 'bp_category_encoded' in df.columns:
            if df['bp_category_encoded'].dtype == 'object':
                print("Converting bp_category_encoded from string to numeric")
                bp_category_mapping = {
                    'Normal': 0,
                    'Elevated': 1,
                    'Hypertension Stage 1': 2,
                    'Hypertension Stage 2': 3,
                    'Hypertensive Crisis': 4
                }
                df['bp_category_numeric'] = df['bp_category_encoded'].map(bp_category_mapping)
                features.append('bp_category_numeric')
            else:
                features.append('bp_category_encoded')
        
        # Create a DataFrame with the combined data
        X_df = pd.DataFrame(X_combined, columns=[f"feature_{i}" for i in range(X_combined.shape[1])])
        
        # Return the preprocessed data and feature names
        return X_df, y_combined, features
        
    except Exception as e:
        print(f"Spark preprocessing failed: {str(e)}")
        print("Falling back to pandas-based preprocessing...")
        
        # Load the data
        df = pd.read_csv(data_path)
        print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("Missing values found:")
            print(missing_values[missing_values > 0])
            # Fill missing values or drop rows with missing values
            df = df.dropna()
            print(f"After dropping rows with missing values: {df.shape[0]} rows")
        
        # Extract features and target
        features = [
            "age", "gender", "height", "weight", "ap_hi", "ap_lo", 
            "cholesterol", "gluc", "smoke", "alco", "active", 
            "age_years", "bmi"
        ]
        
        # Check if bp_category_encoded is a string and needs encoding
        if 'bp_category_encoded' in df.columns:
            # Check the data type
            if df['bp_category_encoded'].dtype == 'object':
                print("Converting bp_category_encoded from string to numeric")
                # Create a mapping for blood pressure categories
                bp_category_mapping = {
                    'Normal': 0,
                    'Elevated': 1,
                    'Hypertension Stage 1': 2,
                    'Hypertension Stage 2': 3,
                    'Hypertensive Crisis': 4
                }
                
                # Apply the mapping and create a new numeric column
                df['bp_category_numeric'] = df['bp_category_encoded'].map(bp_category_mapping)
                
                # Check if any values couldn't be mapped (will be NaN)
                unmapped = df['bp_category_numeric'].isna().sum()
                if unmapped > 0:
                    print(f"Warning: {unmapped} values in bp_category_encoded couldn't be mapped")
                    # Fill NaN values with a default (e.g., most common category)
                    df['bp_category_numeric'].fillna(df['bp_category_numeric'].mode()[0], inplace=True)
                
                # Add the new numeric feature to the features list
                features.append('bp_category_numeric')
                print("Added bp_category_numeric to features list")
            else:
                # If it's already numeric, just add it to the features
                features.append('bp_category_encoded')
        else:
            print("Warning: bp_category_encoded not found in dataset")
        
        # Ensure all required features are in the dataset
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"Warning: The following features are missing from the dataset: {missing_features}")
            features = [f for f in features if f in df.columns]
        
        # Select only the required features and target
        X = df[features]
        y = df['cardio']
        
        # Check class balance
        class_counts = y.value_counts()
        print("Class distribution:")
        print(class_counts)
        
        # Return the preprocessed data
        return X, y, features

def perform_feature_selection(X, y, features, method='rfe', n_features=10):
    """Perform feature selection to identify the most important features."""
    print(f"Performing feature selection using {method} method")
    
    if method == 'rfe':
        # Recursive Feature Elimination
        model = RandomForestClassifier(random_state=RANDOM_STATE)
        selector = RFE(model, n_features_to_select=n_features)
        selector.fit(X, y)
        selected_features = [features[i] for i in range(len(features)) if selector.support_[i]]
    
    elif method == 'model_based':
        # Model-based feature selection
        model = RandomForestClassifier(random_state=RANDOM_STATE)
        model.fit(X, y)
        selector = SelectFromModel(model, prefit=True, threshold='median')
        selected_mask = selector.get_support()
        selected_features = [features[i] for i in range(len(features)) if selected_mask[i]]
    
    else:
        # No feature selection, use all features
        selected_features = features
    
    print(f"Selected {len(selected_features)} features: {selected_features}")
    
    # Plot feature importance if using model-based selection
    if method == 'model_based':
        feature_importance = pd.Series(model.feature_importances_, index=features)
        plt.figure(figsize=(10, 6))
        feature_importance.sort_values(ascending=False).plot(kind='bar')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png')
        plt.show()  # Display the feature importance plot
    
    return selected_features

def train_and_evaluate_models(X, y, features, selected_features=None):
    """Train and evaluate multiple models."""
    if selected_features:
        X = X[selected_features]
        features = selected_features
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Handle class imbalance
    if HAS_SMOTE:
        # Use SMOTE if available
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {X_train_resampled.shape[0]} training samples")
    else:
        # Alternative: Use class weights in the models instead of SMOTE
        print("Using class weights instead of SMOTE for handling class imbalance")
        # Calculate class weights
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}
        print(f"Class weights: {class_weights}")
        
        # No resampling, use original data
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for later use
    joblib.dump(scaler, 'models_improved/scaler.joblib')
    
    # Define models to train - adjust to use class_weight when SMOTE is not available
    models = {
        'logistic_regression': {
            'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, 
                                        class_weight='balanced' if not HAS_SMOTE else None),
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE,
                                           class_weight='balanced' if not HAS_SMOTE else None),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'max_features': ['sqrt', 'log2'],
                'class_weight': ['balanced', None] if HAS_SMOTE else ['balanced']
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9, 1.0]
            }
        },
        'xgboost': {
            'model': xgb.XGBClassifier(random_state=RANDOM_STATE,
                                      scale_pos_weight=class_weights[1]/class_weights[0] if not HAS_SMOTE else 1),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9],
                'scale_pos_weight': [1, 3, 5] if HAS_SMOTE else [class_weights[1]/class_weights[0]]
            }
        }
    }
    
    # Train and evaluate each model
    results = []
    best_models = {}
    
    for model_name, model_info in models.items():
        print(f"\nTraining {model_name}...")
        start_time = time.time()
        
        # Define cross-validation strategy
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        
        # Perform grid search
        grid_search = GridSearchCV(
            model_info['model'],
            model_info['params'],
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train_resampled)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        best_models[model_name] = best_model
        
        # Make predictions
        y_pred = best_model.predict(X_test_scaled)
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Plot and save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Disease'])
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.savefig(f'plots/confusion_matrix_{model_name}.png')
        plt.show()
        plt.close()
        
        # Convert numpy types to Python native types before adding to results
        model_result = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'training_time': float(training_time),
            'best_params': best_model.get_params(),
            'optimal_threshold': float(optimal_threshold)
        }
        
        results.append(model_result)
        
        print(f"{model_name} results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")
        print(f"  Best Parameters: {grid_search.best_params_}")
        print(f"  Training Time: {training_time:.2f} seconds")
        
        # Save the model
        joblib.dump(best_model, f'models_improved/best_model_{model_name}.joblib')
        
        # Create a calibrated version of the model
        calibrated_model = CalibratedClassifierCV(best_model, cv='prefit')
        calibrated_model.fit(X_train_scaled, y_train_resampled)
        joblib.dump(calibrated_model, f'models_improved/calibrated_model_{model_name}.joblib')
    
    # Create an ensemble model (voting classifier)
    print("\nCreating ensemble model...")
    estimators = [(name, model) for name, model in best_models.items()]
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    
    # Calculate ensemble training time
    ensemble_start_time = time.time()
    ensemble.fit(X_train_scaled, y_train_resampled)
    ensemble_training_time = time.time() - ensemble_start_time
    
    # Evaluate ensemble model
    y_pred_ensemble = ensemble.predict(X_test_scaled)
    y_prob_ensemble = ensemble.predict_proba(X_test_scaled)[:, 1]
    
    accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
    precision_ensemble = precision_score(y_test, y_pred_ensemble)
    recall_ensemble = recall_score(y_test, y_pred_ensemble)
    f1_ensemble = f1_score(y_test, y_pred_ensemble)
    auc_ensemble = roc_auc_score(y_test, y_prob_ensemble)
    
    # Find optimal threshold for ensemble
    fpr, tpr, thresholds = roc_curve(y_test, y_prob_ensemble)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold_ensemble = thresholds[optimal_idx]
    
    # Plot and save confusion matrix for ensemble
    cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_ensemble, display_labels=['No Disease', 'Disease'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix - Ensemble')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix_ensemble.png')
    plt.show()
    plt.close()
    
    # Store ensemble results
    ensemble_result = {
        'model_name': 'ensemble',
        'accuracy': float(accuracy_ensemble),
        'precision': float(precision_ensemble),
        'recall': float(recall_ensemble),
        'f1': float(f1_ensemble),
        'auc': float(auc_ensemble),
        'training_time': float(ensemble_training_time),
        'optimal_threshold': float(optimal_threshold_ensemble)
    }
    results.append(ensemble_result)
    
    print("Ensemble model results:")
    print(f"  Accuracy: {accuracy_ensemble:.4f}")
    print(f"  Precision: {precision_ensemble:.4f}")
    print(f"  Recall: {recall_ensemble:.4f}")
    print(f"  F1 Score: {f1_ensemble:.4f}")
    print(f"  AUC: {auc_ensemble:.4f}")
    print(f"  Optimal Threshold: {optimal_threshold_ensemble:.4f}")
    
    # Save the ensemble model
    joblib.dump(ensemble, 'models_improved/ensemble_model.joblib')
    
    # Create a calibrated version of the ensemble model
    calibrated_ensemble = CalibratedClassifierCV(ensemble, cv='prefit')
    calibrated_ensemble.fit(X_train_scaled, y_train_resampled)
    joblib.dump(calibrated_ensemble, 'models_improved/calibrated_model_ensemble.joblib')
    
    # Save feature names
    with open('models_improved/feature_names.json', 'w') as f:
        json.dump(features, f)
    
    # Define the convert_numpy_types function
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(i) for i in obj]
        else:
            return obj
    
    # Save results to JSON
    with open('models_improved/model_metrics.json', 'w') as f:
        json.dump(convert_numpy_types(results), f, indent=2)
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    for result in results:
        model_name = result['model_name']
        if model_name == 'ensemble':
            model = ensemble
        else:
            model = best_models[model_name]
        
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = result['auc']
        
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig('plots/roc_curves.png')
    plt.show()
    
    # Create a combined confusion matrix visualization
    plt.figure(figsize=(15, 10))
    model_names = list(best_models.keys()) + ['ensemble']
    num_models = len(model_names)
    rows = (num_models + 1) // 2
    
    for i, model_name in enumerate(model_names):
        plt.subplot(rows, 2, i+1)
        
        if model_name == 'ensemble':
            cm = confusion_matrix(y_test, y_pred_ensemble)
        else:
            model = best_models[model_name]
            cm = confusion_matrix(y_test, model.predict(X_test_scaled))
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Disease'])
        disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca(), colorbar=False)
        plt.title(f'{model_name}')
    
    plt.tight_layout()
    plt.savefig('plots/all_confusion_matrices.png')
    plt.show()
    plt.close()
    
    return results, best_models, ensemble

def main():
    """Main function to run the training pipeline."""
    print("Starting improved heart disease prediction model training")
    
    # Try multiple possible data paths
    possible_paths = [
        '/kaggle/input/cardiovascular-disease/cardio_data_processed.csv',
        '/kaggle/input/cardiovascular-disease/cardio_train.csv',
        '/kaggle/input/cardiovascular-disease-dataset/cardio_data_processed.csv',
        'data/cardio_data_processed.csv'
    ]
    
    # Try each path until one works
    for data_path in possible_paths:
        try:
            print(f"Attempting to load data from {data_path}")
            X, y, features = load_and_preprocess_data(data_path)
            # If we get here, the data was loaded successfully
            break
        except FileNotFoundError as e:
            print(f"Could not load from {data_path}: {e}")
    else:
        # If we get here, none of the paths worked
        raise FileNotFoundError("Could not find the dataset in any of the expected locations. Please check the dataset path.")
    
    # Perform feature selection
    selected_features = perform_feature_selection(X, y, features, method=None)
    
    # Train and evaluate models
    results, best_models, ensemble = train_and_evaluate_models(X, y, features, selected_features)
    
    # Find the best model based on AUC
    best_result = max(results, key=lambda x: x['auc'])
    
    best_model_name = best_result['model_name']
    
    print(f"\nBest model: {best_model_name}")
    print(f"  AUC: {best_result['auc']:.4f}")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  F1 Score: {best_result['f1']:.4f}")
    
    print("\nTraining completed successfully")

if __name__ == "__main__":
    main()