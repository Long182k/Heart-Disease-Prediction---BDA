#!/usr/bin/env python3
"""
Data Analysis Module for Cardiovascular Disease Prediction

This module provides functions for analyzing and visualizing the cardiovascular
disease dataset to gain insights for feature engineering and modeling.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, isnull
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def create_output_dirs():
    """Create output directories for visualizations and results."""
    os.makedirs("output/visualizations", exist_ok=True)
    os.makedirs("output/models", exist_ok=True)
    os.makedirs("output/results", exist_ok=True)

def load_dataset(file_path="/Users/drake/Documents/UWE/BDA/Heart-Disease-Prediction---BDA/Dataset/cardio_data_processed.csv"):
    """
    Load the cardiovascular disease dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the dataset
        
    Returns:
    --------
    pandas.DataFrame
        The loaded DataFrame
    """
    df = pd.read_csv(file_path)
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def load_spark_dataset(spark, file_path="/Users/drake/Documents/UWE/BDA/Heart-Disease-Prediction---BDA/Dataset/cardio_data_processed.csv"):
    """
    Load the cardiovascular disease dataset into a Spark DataFrame.
    
    Parameters:
    -----------
    spark : pyspark.sql.SparkSession
        The Spark session
    file_path : str
        Path to the dataset
        
    Returns:
    --------
    pyspark.sql.DataFrame
        The loaded Spark DataFrame
    """
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    print(f"Spark DataFrame loaded with {df.count()} rows and {len(df.columns)} columns")
    return df

def explore_data_summary(df):
    """
    Generate summary statistics for the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
        
    Returns:
    --------
    dict
        Dictionary containing summary statistics
    """
    summary = {}
    
    # Basic info
    summary['shape'] = df.shape
    summary['columns'] = df.columns.tolist()
    
    # Missing values
    missing_values = df.isnull().sum()
    summary['missing_values'] = missing_values[missing_values > 0].to_dict()
    
    # Summary statistics for numeric columns
    summary['numeric_stats'] = df.describe().to_dict()
    
    # Class distribution
    summary['class_distribution'] = df['cardio'].value_counts().to_dict()
    
    # Gender distribution
    summary['gender_distribution'] = df['gender'].value_counts().to_dict()
    
    # Age distribution
    summary['age_stats'] = {
        'min': df['age_years'].min(),
        'max': df['age_years'].max(),
        'mean': df['age_years'].mean(),
        'median': df['age_years'].median()
    }
    
    # BMI distribution
    summary['bmi_stats'] = {
        'min': df['bmi'].min(),
        'max': df['bmi'].max(),
        'mean': df['bmi'].mean(),
        'median': df['bmi'].median()
    }
    
    # Blood pressure distribution
    summary['bp_distribution'] = df['bp_category'].value_counts().to_dict()
    
    return summary

def plot_class_distribution(df, output_path="output/visualizations/class_distribution.png"):
    """
    Plot the distribution of the target variable.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    output_path : str
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x='cardio', data=df, palette=['#2ecc71', '#e74c3c'])
    plt.title('Distribution of Cardiovascular Disease', fontsize=16)
    plt.xlabel('Cardiovascular Disease (0 = No, 1 = Yes)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    
    # Add count labels
    counts = df['cardio'].value_counts().sort_index()
    for i, count in enumerate(counts):
        plt.text(i, count + 50, f"{count} ({count/len(df)*100:.1f}%)", 
                 ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_age_distribution(df, output_path="output/visualizations/age_distribution.png"):
    """
    Plot the age distribution by cardiovascular disease status.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    output_path : str
        Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='age_years', hue='cardio', bins=20, 
                 multiple='dodge', palette=['#2ecc71', '#e74c3c'])
    plt.title('Age Distribution by Cardiovascular Disease Status', fontsize=16)
    plt.xlabel('Age (years)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_bmi_distribution(df, output_path="output/visualizations/bmi_distribution.png"):
    """
    Plot the BMI distribution by cardiovascular disease status.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    output_path : str
        Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='bmi', hue='cardio', bins=20, 
                 multiple='dodge', palette=['#2ecc71', '#e74c3c'])
    plt.title('BMI Distribution by Cardiovascular Disease Status', fontsize=16)
    plt.xlabel('BMI (kg/m²)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    
    # Add BMI category lines
    plt.axvline(x=18.5, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=25, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=30, color='gray', linestyle='--', alpha=0.7)
    
    # Add BMI category labels
    plt.text(16, plt.ylim()[1]*0.9, 'Underweight', ha='center', va='center', rotation=90, alpha=0.7)
    plt.text(21.75, plt.ylim()[1]*0.9, 'Normal', ha='center', va='center', rotation=90, alpha=0.7)
    plt.text(27.5, plt.ylim()[1]*0.9, 'Overweight', ha='center', va='center', rotation=90, alpha=0.7)
    plt.text(35, plt.ylim()[1]*0.9, 'Obese', ha='center', va='center', rotation=90, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_bp_distribution(df, output_path="output/visualizations/bp_distribution.png"):
    """
    Plot the blood pressure category distribution by cardiovascular disease status.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    output_path : str
        Path to save the plot
    """
    plt.figure(figsize=(14, 8))
    
    # Create a crosstab of BP category and cardio
    bp_cardio = pd.crosstab(df['bp_category'], df['cardio'], normalize='index') * 100
    
    # Plot stacked bar chart
    bp_cardio.plot(kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'], figsize=(14, 8))
    plt.title('Blood Pressure Category vs. Cardiovascular Disease', fontsize=16)
    plt.xlabel('Blood Pressure Category', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(['No Disease', 'Disease'], title='Cardiovascular Disease')
    
    # Add count labels
    counts = df['bp_category'].value_counts().sort_index()
    for i, (category, count) in enumerate(counts.items()):
        plt.text(i, 105, f"n={count}", ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_matrix(df, output_path="output/visualizations/correlation_matrix.png"):
    """
    Plot the correlation matrix of numeric features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    output_path : str
        Path to save the plot
    """
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Plot heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Numeric Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_categorical_features(df, output_path_prefix="output/visualizations/"):
    """
    Plot the distribution of categorical features by cardiovascular disease status.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    output_path_prefix : str
        Prefix for the output paths
    """
    categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        
        # Create a crosstab of feature and cardio
        feature_cardio = pd.crosstab(df[feature], df['cardio'], normalize='index') * 100
        
        # Plot stacked bar chart
        feature_cardio.plot(kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'])
        plt.title(f'{feature.capitalize()} vs. Cardiovascular Disease', fontsize=16)
        plt.xlabel(feature.capitalize(), fontsize=14)
        plt.ylabel('Percentage (%)', fontsize=14)
        plt.legend(['No Disease', 'Disease'], title='Cardiovascular Disease')
        
        # Add count labels
        counts = df[feature].value_counts().sort_index()
        for i, (category, count) in enumerate(counts.items()):
            plt.text(i, 105, f"n={count}", ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{output_path_prefix}{feature}_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

def plot_bp_components(df, output_path="output/visualizations/bp_components.png"):
    """
    Plot the systolic and diastolic blood pressure by cardiovascular disease status.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    output_path : str
        Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot
    sns.scatterplot(x='ap_hi', y='ap_lo', hue='cardio', data=df, 
                    palette=['#2ecc71', '#e74c3c'], alpha=0.6)
    
    # Add normal blood pressure region
    plt.axvspan(90, 120, alpha=0.2, color='green')
    plt.axhspan(60, 80, alpha=0.2, color='green')
    
    # Add labels for blood pressure categories
    plt.text(105, 55, 'Normal BP\n(90-120/60-80)', ha='center', va='center', 
             bbox=dict(facecolor='white', alpha=0.7))
    plt.text(130, 85, 'Hypertension\n(>130/>80)', ha='center', va='center', 
             bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title('Systolic vs. Diastolic Blood Pressure by Disease Status', fontsize=16)
    plt.xlabel('Systolic Blood Pressure (mmHg)', fontsize=14)
    plt.ylabel('Diastolic Blood Pressure (mmHg)', fontsize=14)
    plt.legend(title='Cardiovascular Disease', labels=['No', 'Yes'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_age_gender_distribution(df, output_path="output/visualizations/age_gender_distribution.png"):
    """
    Plot the age distribution by gender and cardiovascular disease status.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    output_path : str
        Path to save the plot
    """
    plt.figure(figsize=(14, 8))
    
    # Map gender values to labels
    df_plot = df.copy()
    df_plot['gender_label'] = df_plot['gender'].map({1: 'Female', 2: 'Male'})
    
    # Create violin plot
    sns.violinplot(x='gender_label', y='age_years', hue='cardio', 
                   data=df_plot, palette=['#2ecc71', '#e74c3c'], split=True)
    
    plt.title('Age Distribution by Gender and Cardiovascular Disease Status', fontsize=16)
    plt.xlabel('Gender', fontsize=14)
    plt.ylabel('Age (years)', fontsize=14)
    plt.legend(title='Cardiovascular Disease', labels=['No', 'Yes'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_risk_factors(df, output_path="output/visualizations/risk_factors.png"):
    """
    Plot the impact of multiple risk factors on cardiovascular disease.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    output_path : str
        Path to save the plot
    """
    # Create a risk factor count
    df_plot = df.copy()
    
    # Convert cholesterol and glucose to binary risk factors (1 if > 1, else 0)
    df_plot['chol_risk'] = (df_plot['cholesterol'] > 1).astype(int)
    df_plot['gluc_risk'] = (df_plot['gluc'] > 1).astype(int)
    
    # Create inactive column (inverse of active)
    df_plot['inactive'] = 1 - df_plot['active']
    
    # Sum risk factors
    risk_factors = ['chol_risk', 'gluc_risk', 'smoke', 'alco', 'inactive']
    df_plot['risk_factor_count'] = df_plot[risk_factors].sum(axis=1)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Calculate percentage of disease by risk factor count
    risk_disease = pd.crosstab(df_plot['risk_factor_count'], df_plot['cardio'], normalize='index') * 100
    
    # Plot bar chart
    ax = risk_disease[1].plot(kind='bar', color='#e74c3c')
    
    plt.title('Cardiovascular Disease Prevalence by Number of Risk Factors', fontsize=16)
    plt.xlabel('Number of Risk Factors', fontsize=14)
    plt.ylabel('Percentage with Cardiovascular Disease (%)', fontsize=14)
    
    # Add count labels
    counts = df_plot['risk_factor_count'].value_counts().sort_index()
    for i, (count_val, count) in enumerate(counts.items()):
        plt.text(i, risk_disease[1][count_val] + 2, f"n={count}", ha='center', va='bottom', fontsize=12)
    
    # Add horizontal line for average prevalence
    avg_prevalence = df_plot['cardio'].mean() * 100
    plt.axhline(y=avg_prevalence, color='gray', linestyle='--', alpha=0.7)
    plt.text(len(counts)-1, avg_prevalence + 2, f'Average: {avg_prevalence:.1f}%', 
             ha='right', va='bottom', color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_dataset(file_path="/Users/drake/Documents/UWE/BDA/Heart-Disease-Prediction---BDA/Dataset/cardio_data_processed.csv"):
    """
    Perform a comprehensive analysis of the dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the dataset
    """
    # Create output directories
    create_output_dirs()
    
    # Load dataset
    df = load_dataset(file_path)
    
    # Generate summary statistics
    summary = explore_data_summary(df)
    
    # Print key findings
    print("\n=== Dataset Summary ===")
    print(f"Total records: {summary['shape'][0]}")
    print(f"Features: {summary['shape'][1]}")
    print(f"\nClass distribution:")
    for label, count in summary['class_distribution'].items():
        print(f"  - Class {label}: {count} ({count/summary['shape'][0]*100:.1f}%)")
    
    print(f"\nAge range: {summary['age_stats']['min']:.1f} - {summary['age_stats']['max']:.1f} years")
    print(f"Mean age: {summary['age_stats']['mean']:.1f} years")
    
    print(f"\nBMI range: {summary['bmi_stats']['min']:.1f} - {summary['bmi_stats']['max']:.1f} kg/m²")
    print(f"Mean BMI: {summary['bmi_stats']['mean']:.1f} kg/m²")
    
    print(f"\nBlood pressure distribution:")
    for category, count in summary['bp_distribution'].items():
        print(f"  - {category}: {count} ({count/summary['shape'][0]*100:.1f}%)")
    
    # Generate visualizations
    print("\n=== Generating Visualizations ===")
    
    print("Plotting class distribution...")
    plot_class_distribution(df)
    
    print("Plotting age distribution...")
    plot_age_distribution(df)
    
    print("Plotting BMI distribution...")
    plot_bmi_distribution(df)
    
    print("Plotting blood pressure distribution...")
    plot_bp_distribution(df)
    
    print("Plotting correlation matrix...")
    plot_correlation_matrix(df)
    
    print("Plotting categorical features...")
    plot_categorical_features(df)
    
    print("Plotting blood pressure components...")
    plot_bp_components(df)
    
    print("Plotting age and gender distribution...")
    plot_age_gender_distribution(df)
    
    print("Plotting risk factors analysis...")
    plot_risk_factors(df)
    
    print("\nAnalysis complete. Visualizations saved to 'output/visualizations/'")

if __name__ == "__main__":
    analyze_dataset()