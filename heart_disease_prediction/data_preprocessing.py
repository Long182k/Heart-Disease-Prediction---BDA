#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
import pandas as pd
import numpy as np
import os
import sys

def create_spark_session(app_name="HeartDiseasePredictor"):
    """Create and return a Spark session."""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.ui.port", "4050") \
        .config("spark.local.dir", "/tmp/spark-temp") \
        .getOrCreate()

def load_data(spark, data_path):
    """Load the heart disease dataset into a Spark DataFrame."""
    print(f"Loading data from {data_path}...")
    return spark.read.csv(data_path, header=True, inferSchema=True)

def preprocess_data(df):
    """Preprocess the data: handle missing values, convert categorical features, etc."""
    print("Preprocessing data...")
    
    # Print schema
    print("Original Schema:")
    df.printSchema()
    
    # Check for missing values
    print("Checking for missing values:")
    for col_name in df.columns:
        missing_count = df.filter(df[col_name].isNull()).count()
        if missing_count > 0:
            print(f"Column {col_name} has {missing_count} missing values.")
    
    # List of numerical features
    numerical_features = [col_name for col_name in df.columns 
                         if col_name != 'target' and df.select(col_name).dtypes[0][1] != 'string']
    
    # Impute missing values in numerical features
    imputer = Imputer(
        inputCols=numerical_features,
        outputCols=[f"{col_name}_imputed" for col_name in numerical_features]
    ).setStrategy("mean")
    
    # Assemble features into a vector
    assembled_features = [f"{col_name}_imputed" for col_name in numerical_features]
    assembler = VectorAssembler(inputCols=assembled_features, outputCol="features_unscaled")
    
    # Scale features
    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features")
    
    # Create preprocessing pipeline
    preprocessing_pipeline = Pipeline(stages=[imputer, assembler, scaler])
    
    # Fit preprocessing pipeline
    preprocessing_model = preprocessing_pipeline.fit(df)
    processed_df = preprocessing_model.transform(df)
    
    # Select relevant columns
    final_df = processed_df.select("target", "features")
    
    # Show some processed data
    print("Processed Data Sample:")
    final_df.show(5)
    
    return final_df, preprocessing_model

def handle_imbalanced_data(df):
    """Handle imbalanced data using undersampling or oversampling."""
    print("Handling imbalanced data...")
    
    # Check class distribution
    class_counts = df.groupBy("target").count().orderBy("target")
    class_counts.show()
    
    # Convert to pandas to apply SMOTE (in Spark, we would need to implement it manually)
    pandas_df = df.toPandas()
    
    # For simplicity, just doing random undersampling here
    # In a real project, SMOTE would be better (using imbalanced-learn library)
    class_0 = pandas_df[pandas_df['target'] == 0]
    class_1 = pandas_df[pandas_df['target'] == 1]
    
    # Get the minority class count
    min_class_count = min(len(class_0), len(class_1))
    
    # Undersample the majority class
    if len(class_0) > len(class_1):
        class_0 = class_0.sample(min_class_count, random_state=42)
    else:
        class_1 = class_1.sample(min_class_count, random_state=42)
    
    # Combine the balanced classes
    balanced_df = pd.concat([class_0, class_1])
    
    # Convert back to Spark DataFrame
    balanced_spark_df = spark.createDataFrame(balanced_df)
    
    print("Balanced class distribution:")
    balanced_spark_df.groupBy("target").count().orderBy("target").show()
    
    return balanced_spark_df

def prepare_train_test_data(df, test_ratio=0.2):
    """Split the dataset into training and testing sets."""
    print(f"Splitting data into training ({1-test_ratio:.0%}) and testing ({test_ratio:.0%}) sets...")
    train_df, test_df = df.randomSplit([1 - test_ratio, test_ratio], seed=42)
    
    print(f"Training set size: {train_df.count()}")
    print(f"Testing set size: {test_df.count()}")
    
    return train_df, test_df

def save_processed_data(train_df, test_df, output_dir="processed_data"):
    """Save the processed data to disk."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert to pandas and save as CSV for easier handling
    train_pandas = train_df.toPandas()
    test_pandas = test_df.toPandas()
    
    # Save to CSV
    train_pandas.to_csv(f"{output_dir}/train_data.csv", index=False)
    test_pandas.to_csv(f"{output_dir}/test_data.csv", index=False)
    
    print(f"Saved processed data to {output_dir}/")

if __name__ == "__main__":
    # Create Spark session
    spark = create_spark_session()
    
    # Set path to dataset
    data_path = "../Dataset/archive/heart_disease.csv"
    
    try:
        # Load data
        heart_df = load_data(spark, data_path)
        
        # Preprocess data
        processed_df, preprocessing_model = preprocess_data(heart_df)
        
        # Handle imbalanced data
        balanced_df = handle_imbalanced_data(processed_df)
        
        # Split data into training and testing sets
        train_df, test_df = prepare_train_test_data(balanced_df)
        
        # Save processed data
        save_processed_data(train_df, test_df, "../processed_data")
        
        print("Data preprocessing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        # Stop Spark session
        spark.stop()
