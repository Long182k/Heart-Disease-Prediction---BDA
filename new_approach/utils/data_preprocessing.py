#!/usr/bin/env python3
"""
Data Preprocessing Module for Cardiovascular Disease Prediction

This module provides functions for loading, cleaning, exploring, and splitting
the cardiovascular disease dataset.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, count

def load_data(file_path):
    """
    Load the cardiovascular disease dataset from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded dataset
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df

def clean_data(df):
    """
    Clean the cardiovascular disease dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to clean
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataset
    """
    print("Cleaning data...")
    
    # Make a copy to avoid modifying the original dataframe
    df_clean = df.copy()
    
    # Drop duplicates
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    print(f"Removed {initial_rows - df_clean.shape[0]} duplicate rows")
    
    # Always define numeric and categorical columns here
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns

    # Check for missing values
    missing_values = df_clean.isnull().sum()
    if missing_values.sum() > 0:
        print("Missing values found:")
        print(missing_values[missing_values > 0])
        
        # Fill missing values or drop rows with missing values
        # For numerical columns, fill with median
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # For categorical columns, fill with mode
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    else:
        print("No missing values found")
    
    # Check for outliers in numerical columns
    print("Checking for outliers...")
    for col in numeric_cols:
        if col not in ['id', 'cardio']:  # Skip ID and target columns
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
            if not outliers.empty:
                print(f"Found {len(outliers)} outliers in column '{col}'")
                
                # Option 1: Cap outliers at the bounds
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                
                # Option 2: Remove outliers (commented out)
                # df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    # Ensure data types are appropriate
    print("Ensuring appropriate data types...")
    
    # Convert categorical columns to category type if needed
    categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bp_category']
    for col in categorical_features:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype('category')
    
    print(f"Cleaned dataset shape: {df_clean.shape}")
    return df_clean

def explore_data(df):
    """
    Explore the cardiovascular disease dataset and generate visualizations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to explore
    """
    print("Exploring data...")
    
    # Create output directory for visualizations
    os.makedirs("output/visualizations", exist_ok=True)
    
    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe())
    
    # Data types and non-null counts
    print("\nData types and non-null counts:")
    print(df.info())
    
    # Target distribution
    print("\nTarget distribution:")
    target_counts = df['cardio'].value_counts()
    print(target_counts)
    
    plt.figure(figsize=(8, 6))
    sns.countplot(x='cardio', data=df)
    plt.title('Distribution of Cardiovascular Disease')
    plt.xlabel('Cardiovascular Disease (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    plt.savefig("output/visualizations/target_distribution.png")
    
    # Age distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='age_years', hue='cardio', multiple='stack', bins=20)
    plt.title('Age Distribution by Cardiovascular Disease Status')
    plt.xlabel('Age (years)')
    plt.ylabel('Count')
    plt.savefig("output/visualizations/age_distribution.png")
    
    # BMI distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='bmi', hue='cardio', multiple='stack', bins=20)
    plt.title('BMI Distribution by Cardiovascular Disease Status')
    plt.xlabel('BMI')
    plt.ylabel('Count')
    plt.savefig("output/visualizations/bmi_distribution.png")
    
    # Blood pressure categories
    plt.figure(figsize=(12, 6))
    sns.countplot(x='bp_category', hue='cardio', data=df, order=df['bp_category'].value_counts().index)
    plt.title('Blood Pressure Categories by Cardiovascular Disease Status')
    plt.xlabel('Blood Pressure Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig("output/visualizations/bp_categories.png")
    
    # Correlation matrix
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Numerical Features')
    plt.savefig("output/visualizations/correlation_matrix.png")
    
    # Categorical features analysis
    categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(categorical_features):
        if feature in df.columns:
            sns.countplot(x=feature, hue='cardio', data=df, ax=axes[i])
            axes[i].set_title(f'{feature} by Cardiovascular Disease Status')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig("output/visualizations/categorical_features.png")
    
    print("Visualizations saved to output/visualizations/")

def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split the dataset into training, validation, and test sets.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to split
    test_size : float
        Proportion of the dataset to include in the test split
    val_size : float
        Proportion of the dataset to include in the validation split
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (train_df, val_df, test_df)
    """
    print("Splitting data into train, validation, and test sets...")
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['cardio']
    )
    
    # Second split: separate validation set from training set
    # Adjust validation size to account for the reduced dataset size
    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size_adjusted, 
        random_state=random_state, stratify=train_val_df['cardio']
    )
    
    print(f"Train set shape: {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    return train_df, val_df, test_df

def load_spark_data(spark, file_path):
    """
    Load the cardiovascular disease dataset into a Spark DataFrame.
    
    Parameters:
    -----------
    spark : pyspark.sql.SparkSession
        Spark session
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pyspark.sql.DataFrame
        Loaded dataset as a Spark DataFrame
    """
    print(f"Loading data from {file_path} into Spark DataFrame...")
    spark_df = spark.read.csv(file_path, header=True, inferSchema=True)
    print(f"Loaded Spark DataFrame with {spark_df.count()} rows and {len(spark_df.columns)} columns")
    return spark_df

def clean_spark_data(spark_df):
    """
    Clean the cardiovascular disease dataset in Spark.
    
    Parameters:
    -----------
    spark_df : pyspark.sql.DataFrame
        Spark DataFrame to clean
        
    Returns:
    --------
    pyspark.sql.DataFrame
        Cleaned Spark DataFrame
    """
    print("Cleaning Spark DataFrame...")
    
    # Drop duplicates
    initial_count = spark_df.count()
    spark_df_clean = spark_df.dropDuplicates()
    print(f"Removed {initial_count - spark_df_clean.count()} duplicate rows")
    
    # Check for missing values
    missing_counts = spark_df_clean.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in spark_df_clean.columns])
    missing_counts.show()
    
    # Handle missing values if any
    for column in spark_df_clean.columns:
        # For numeric columns, fill with median
        if spark_df_clean.schema[column].dataType.simpleString() in ['int', 'double', 'float']:
            median_value = spark_df_clean.approxQuantile(column, [0.5], 0.01)[0]
            spark_df_clean = spark_df_clean.fillna(median_value, subset=[column])
    
    # Convert categorical columns to appropriate types
    categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bp_category']
    for col_name in categorical_features:
        if col_name in spark_df_clean.columns:
            spark_df_clean = spark_df_clean.withColumn(col_name, spark_df_clean[col_name].cast("string"))
    
    print(f"Cleaned Spark DataFrame has {spark_df_clean.count()} rows")
    return spark_df_clean

def split_spark_data(spark_df, test_size=0.2, val_size=0.1, seed=42):
    """
    Split the Spark DataFrame into training, validation, and test sets.
    
    Parameters:
    -----------
    spark_df : pyspark.sql.DataFrame
        Spark DataFrame to split
    test_size : float
        Proportion of the dataset to include in the test split
    val_size : float
        Proportion of the dataset to include in the validation split
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (train_df, val_df, test_df) as Spark DataFrames
    """
    print("Splitting Spark DataFrame into train, validation, and test sets...")
    
    # Calculate split weights
    train_weight = 1 - test_size - val_size
    
    # Split the data
    train_df, val_df, test_df = spark_df.randomSplit([train_weight, val_size, test_size], seed=seed)
    
    print(f"Train set count: {train_df.count()}")
    print(f"Validation set count: {val_df.count()}")
    print(f"Test set count: {test_df.count()}")
    
    return train_df, val_df, test_df