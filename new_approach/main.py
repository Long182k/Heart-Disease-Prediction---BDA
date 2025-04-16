import os
import argparse
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import json
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from utils.data_preprocessing import load_data, clean_data, explore_data
from feature_engineering import create_medical_features, encode_categorical_features, select_features
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Cardiovascular Disease Prediction (XGBoost API-ready)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    parser.add_argument('--explore', action='store_true', help='Perform exploratory data analysis')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model')
    return parser.parse_args()

def setup_directories(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)

def run_classification_models_workflow(
    df, output_dir, explore=False, save_model=True
):
    df_clean = clean_data(df)
    if explore:
        explore_data(df_clean)
    
    # Define the 13 required features + target
    features = [
        'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
        'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'bp_category'
    ]
    features_with_target = features + ['cardio']
    
    # Ensure we have all required features
    df_selected = df_clean[features_with_target].copy()
    
    # If bmi is missing, calculate it
    if df_selected['bmi'].isnull().any():
        df_selected['bmi'] = df_selected['weight'] / ((df_selected['height'] / 100) ** 2)
    
    # If bp_category is missing, calculate it
    if 'bp_category' not in df_selected.columns or df_selected['bp_category'].isnull().any():
        from model_deployment import categorize_blood_pressure
        df_selected['bp_category'] = df_selected.apply(
            lambda row: categorize_blood_pressure(row['ap_hi'], row['ap_lo']), axis=1
        )
    
    # Encode categorical features if needed
    if df_selected['bp_category'].dtype == 'object':
        df_selected = encode_categorical_features(df_selected)
    
    # Continue with the rest of the workflow...
    # ...

    # Optionally encode bp_category as integer if needed by your model
    # from sklearn.preprocessing import LabelEncoder
    # le = LabelEncoder()
    # df_selected['bp_category'] = le.fit_transform(df_selected['bp_category'])

    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df_selected, test_size=0.2, random_state=42, stratify=df_selected['cardio'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['cardio'])

    # Prepare features and target
    X_train = train_df.drop('cardio', axis=1)
    print("X_train.columns.tolist()",X_train.columns.tolist())

    y_train = train_df['cardio']
    X_val = val_df.drop('cardio', axis=1)
    y_val = val_df['cardio']
    X_test = test_df.drop('cardio', axis=1)
    y_test = test_df['cardio']

    # Remove ID column if present
    if 'id' in X_train.columns:
        X_train = X_train.drop('id', axis=1)
        X_val = X_val.drop('id', axis=1)
        X_test = X_test.drop('id', axis=1)

    # Drop non-numeric columns (e.g., original categorical columns)
    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    # Train and evaluate models
    results = {}
    models = [
        {
            "model_name": "logistic_regression",
            "estimator": LogisticRegression(
                C=1.0, penalty='l2', solver='liblinear', random_state=42, max_iter=1000
            ),
            "params": {"C": 1.0, "penalty": "l2", "solver": "liblinear"}
        },
        {
            "model_name": "random_forest",
            "estimator": RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=2, random_state=42
            ),
            "params": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2}
        },
        {
            "model_name": "gradient_boosting",
            "estimator": GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            ),
            "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5}
        }
    ]

    metrics_list = []
    best_model = None
    best_f1 = -1
    best_model_name = ""
    best_model_obj = None

    for model_info in models:
        print(f"\n--- Training {model_info['model_name']} ---")
        start_time = time.time()
        model = model_info["estimator"]
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

        # === Confusion Matrix Visualization and Saving ===
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {model_info["model_name"]}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_path = os.path.join(output_dir, "results", f'{model_info["model_name"]}_confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion matrix saved to {cm_path}")

        # Save confusion matrix as CSV
        cm_csv_path = os.path.join(output_dir, "results", f'{model_info["model_name"]}_confusion_matrix.csv')
        pd.DataFrame(cm).to_csv(cm_csv_path, index=False)
        print(f"Confusion matrix CSV saved to {cm_csv_path}")

        # === ROC Curve Visualization and Saving ===
        if y_pred_proba is not None:
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve: {model_info["model_name"]}')
            plt.legend(loc='lower right')
            roc_path = os.path.join(output_dir, "results", f'{model_info["model_name"]}_roc_curve.png')
            plt.savefig(roc_path)
            plt.close()
            print(f"ROC curve saved to {roc_path}")

            # Save ROC curve data as CSV
            roc_csv_path = os.path.join(output_dir, "results", f'{model_info["model_name"]}_roc_curve.csv')
            pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}).to_csv(roc_csv_path, index=False)
            print(f"ROC curve data saved to {roc_csv_path}")

        # === Classification Report Printing and Saving ===
        report = classification_report(y_test, y_pred, output_dict=True)
        report_txt = classification_report(y_test, y_pred)
        print("\nClassification Report:")
        print(report_txt)
        report_path = os.path.join(output_dir, "results", f'{model_info["model_name"]}_classification_report.txt')
        with open(report_path, "w") as f:
            f.write(report_txt)
        print(f"Classification report saved to {report_path}")

        # Save classification report as JSON
        report_json_path = os.path.join(output_dir, "results", f'{model_info["model_name"]}_classification_report.json')
        with open(report_json_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Classification report JSON saved to {report_json_path}")

        metrics = {
            "model_name": model_info["model_name"],
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "training_time": training_time,
            "best_params": model_info["params"]
        }
        metrics_list.append(metrics)

        print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = model_info["model_name"]
            best_model_obj = model

    # Save metrics to JSON
    metrics_path = os.path.join(output_dir, "results", "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    print(f"\nModel metrics saved to {metrics_path}")

    # Save best model as .pkl
    if save_model:
        model_path = os.path.join(output_dir, "models", f"{best_model_name}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_model_obj, f)
        print(f"Best model ({best_model_name}) saved as .pkl to {model_path}")

    return best_model_obj

def main():
    args = parse_arguments()
    setup_directories(args.output_dir)
    df = load_data(args.data_path)
    run_classification_models_workflow(
        df,
        output_dir=args.output_dir,
        explore=args.explore,
        save_model=args.save_model
    )
    print("\n=== Workflow completed successfully ===\n")

if __name__ == "__main__":
    main()