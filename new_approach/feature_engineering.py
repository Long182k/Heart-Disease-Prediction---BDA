#!/usr/bin/env python3
"""
Feature Engineering Module for Cardiovascular Disease Prediction

This module provides functions for creating, transforming, and selecting features
for cardiovascular disease prediction.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkStandardScaler
from pyspark.ml.feature import OneHotEncoder as SparkOneHotEncoder
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml import Pipeline as SparkPipeline

def create_medical_features(df):
    """
    Create additional medical features from existing data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with original features
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with additional medical features
    """
    print("Creating additional medical features...")
    
    # Make a copy to avoid modifying the original dataframe
    df_features = df.copy()
    
    # Create pulse pressure (difference between systolic and diastolic)
    if 'ap_hi' in df_features.columns and 'ap_lo' in df_features.columns:
        df_features['pulse_pressure'] = df_features['ap_hi'] - df_features['ap_lo']
        print("Created pulse pressure feature")
    
    # Create mean arterial pressure (MAP)
    if 'ap_hi' in df_features.columns and 'ap_lo' in df_features.columns:
        df_features['mean_arterial_pressure'] = (df_features['ap_hi'] + 2 * df_features['ap_lo']) / 3
        print("Created mean arterial pressure feature")
    
    # Create BMI categories
    if 'bmi' in df_features.columns:
        df_features['bmi_category'] = pd.cut(
            df_features['bmi'],
            bins=[0, 18.5, 25, 30, 35, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese Class I', 'Obese Class II-III']
        )
        print("Created BMI category feature")
    
    # Create age groups
    if 'age_years' in df_features.columns:
        df_features['age_group'] = pd.cut(
            df_features['age_years'],
            bins=[0, 40, 50, 60, 100],
            labels=['<40', '40-50', '50-60', '>60']
        )
        print("Created age group feature")
    
    # Create combined risk factors count
    risk_factors = ['cholesterol', 'gluc', 'smoke', 'alco']
    if all(factor in df_features.columns for factor in risk_factors):
        # Convert cholesterol and glucose to numeric if they are categorical
        if pd.api.types.is_categorical_dtype(df_features['cholesterol']):
            df_features['cholesterol'] = df_features['cholesterol'].astype(int)
        if pd.api.types.is_categorical_dtype(df_features['gluc']):
            df_features['gluc'] = df_features['gluc'].astype(int)
        # Convert smoke and alco to int if they are categorical
        if pd.api.types.is_categorical_dtype(df_features['smoke']):
            df_features['smoke'] = df_features['smoke'].astype(int)
        if pd.api.types.is_categorical_dtype(df_features['alco']):
            df_features['alco'] = df_features['alco'].astype(int)
        df_features['cholesterol_elevated'] = (df_features['cholesterol'] > 1).astype(int)
        df_features['glucose_elevated'] = (df_features['gluc'] > 1).astype(int)
        
        df_features['risk_factors_count'] = (
            df_features['cholesterol_elevated'] +
            df_features['glucose_elevated'] +
            df_features['smoke'] +
            df_features['alco']
        )
        print("Created risk factors count feature")
    
    # Create BP and lifestyle interaction features
    if 'bp_category' in df_features.columns and 'active' in df_features.columns:
        # Create a feature for hypertensive patients who are inactive
        hypertension_cols = ['Hypertension Stage 1', 'Hypertension Stage 2']
        df_features['hypertensive_inactive'] = (
            (df_features['bp_category'].isin(hypertension_cols)) & 
            (df_features['active'] == 0)
        ).astype(int)
        print("Created hypertensive inactive feature")
    
    print(f"Created dataset with {df_features.shape[1]} features")
    return df_features

def encode_categorical_features(df):
    """
    Encode categorical features in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with categorical features
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with encoded categorical features
    """
    print("Encoding categorical features...")
    
    # Make a copy to avoid modifying the original dataframe
    df_encoded = df.copy()
    
    # Identify categorical columns
    categorical_cols = df_encoded.select_dtypes(include=['category', 'object']).columns
    
    if len(categorical_cols) > 0:
        print(f"Found {len(categorical_cols)} categorical columns: {list(categorical_cols)}")
        
        # One-hot encode categorical columns
        for col in categorical_cols:
            # Skip encoding if the column is already encoded or is the target
            if col == 'cardio' or col.endswith('_encoded'):
                continue
                
            # Get dummies (one-hot encoding)
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False)
            
            # Add dummy columns to the dataframe
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            
            # Keep the original categorical column for interpretability
            # df_encoded = df_encoded.drop(col, axis=1)
            
            print(f"Encoded {col} into {dummies.shape[1]} dummy variables")
    else:
        print("No categorical columns found for encoding")
    
    print(f"Encoded dataset has {df_encoded.shape[1]} features")
    return df_encoded

def select_features(df, target_col='cardio', method='f_classif', k=10):
    """
    Select the most important features for prediction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with features
    target_col : str
        Name of the target column
    method : str
        Feature selection method ('f_classif', 'mutual_info', 'rfe')
    k : int
        Number of features to select
        
    Returns:
    --------
    tuple
        (selected_features, df_selected)
    """
    print(f"Selecting top {k} features using {method}...")
    
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Remove non-numeric columns and columns with constant values
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X_numeric = X[numeric_cols]
    
    # Remove ID column if present
    if 'id' in X_numeric.columns:
        X_numeric = X_numeric.drop('id', axis=1)
    
    # Check for constant columns
    constant_cols = [col for col in X_numeric.columns if X_numeric[col].nunique() == 1]
    if constant_cols:
        print(f"Removing {len(constant_cols)} constant columns: {constant_cols}")
        X_numeric = X_numeric.drop(constant_cols, axis=1)
    
    # Select features based on the specified method
    if method == 'f_classif':
        selector = SelectKBest(f_classif, k=min(k, X_numeric.shape[1]))
        X_new = selector.fit_transform(X_numeric, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X_numeric.columns[selected_mask].tolist()
        
    elif method == 'mutual_info':
        selector = SelectKBest(mutual_info_classif, k=min(k, X_numeric.shape[1]))
        X_new = selector.fit_transform(X_numeric, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X_numeric.columns[selected_mask].tolist()
        
    elif method == 'rfe':
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=min(k, X_numeric.shape[1]), step=1)
        X_new = selector.fit_transform(X_numeric, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X_numeric.columns[selected_mask].tolist()
        
    else:
        raise ValueError(f"Unsupported feature selection method: {method}")
    
    # Create a new dataframe with selected features and target
    df_selected = pd.concat([X[selected_features], df[target_col]], axis=1)
    
    print(f"Selected {len(selected_features)} features: {selected_features}")
    return selected_features, df_selected

def create_feature_pipeline(categorical_cols, numeric_cols):
    """
    Create a scikit-learn pipeline for feature preprocessing.
    
    Parameters:
    -----------
    categorical_cols : list
        List of categorical column names
    numeric_cols : list
        List of numeric column names
        
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Feature preprocessing pipeline
    """
    print("Creating feature preprocessing pipeline...")
    
    # Define preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Define preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Create the preprocessing pipeline
    feature_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    
    print("Feature pipeline created")
    return feature_pipeline

def create_spark_feature_pipeline(categorical_cols, numeric_cols):
    """
    Create a Spark ML pipeline for feature preprocessing.
    
    Parameters:
    -----------
    categorical_cols : list
        List of categorical column names
    numeric_cols : list
        List of numeric column names
        
    Returns:
    --------
    pyspark.ml.Pipeline
        Spark feature preprocessing pipeline
    """
    print("Creating Spark feature preprocessing pipeline...")
    
    # Create stages list for the pipeline
    stages = []
    
    # Process categorical columns
    if categorical_cols:
        # String indexers for categorical columns
        indexers = [
            StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
            for col in categorical_cols
        ]
        stages.extend(indexers)
        
        # One-hot encode indexed categorical columns
        encoder = SparkOneHotEncoder(
            inputCols=[f"{col}_idx" for col in categorical_cols],
            outputCols=[f"{col}_vec" for col in categorical_cols]
        )
        stages.append(encoder)
    
    # Combine all features into a single vector column
    feature_cols = [f"{col}_vec" for col in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
    stages.append(assembler)
    
    # Scale features
    scaler = SparkStandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
    stages.append(scaler)
    
    # Create the pipeline
    pipeline = SparkPipeline(stages=stages)
    
    print("Spark feature pipeline created")
    return pipeline