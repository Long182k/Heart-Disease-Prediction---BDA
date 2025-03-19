import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, stddev
from pyspark.ml.feature import VectorAssembler, StandardScaler
import os
import sys
import numpy as np

def create_spark_session(app_name="HeartDiseasePreprocessing"):
    """Create and return a Spark session."""
    # Since you're running in WSL, we need to use the WSL Java path
    if "JAVA_HOME" not in os.environ:
        # Use the WSL Java path
        os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
        print(f"Set JAVA_HOME to {os.environ['JAVA_HOME']}")
    
    # Configure Spark to use the correct Python executable
    os.environ["PYSPARK_PYTHON"] = sys.executable
    
    # Create and return the Spark session
    print("Initializing Spark session...")
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .master("local[*]") \
        .getOrCreate()

def preprocess_data(file_path):
    """
    Preprocess the cardio dataset using Spark for distributed processing
    and return train/test splits for modeling.
    """
    try:
        # Create Spark session
        spark = create_spark_session()
        spark.sparkContext.setLogLevel("ERROR")  # Reduce log verbosity
        
        print("Successfully created Spark session")
        
        # Load data directly into Spark DataFrame
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        print(f"Loaded data with {df.count()} rows and {len(df.columns)} columns")
        
        # Handle missing values - replace with median for numeric columns
        numeric_cols = [field.name for field in df.schema.fields 
                       if field.dataType.simpleString() in ['double', 'int', 'float'] 
                       and field.name not in ['id', 'cardio']]
        
        # Calculate medians for each numeric column
        for col_name in numeric_cols:
            median_value = df.approxQuantile(col_name, [0.5], 0.001)[0]
            df = df.withColumn(col_name, when(col(col_name).isNull(), median_value).otherwise(col(col_name)))
        
        # Prepare features for ML - exclude id, target column, and text columns
        feature_cols = [col for col in df.columns if col not in ['id', 'cardio', 'bp_category']]
        print(f"Using {len(feature_cols)} features: {', '.join(feature_cols[:5])}...")
        
        # Create feature vector
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembled_features")
        assembled_df = assembler.transform(df)
        
        # Scale features using Spark's StandardScaler
        scaler = StandardScaler(inputCol="assembled_features", outputCol="features", 
                               withStd=True, withMean=True)
        scaler_model = scaler.fit(assembled_df)
        processed_df = scaler_model.transform(assembled_df)
        
        # Convert back to Pandas for SMOTE and train/test split
        pandas_processed = processed_df.select("features", "cardio").toPandas()
        
        # Extract features and target
        X = np.array([x.toArray() for x in pandas_processed["features"]])
        y = pandas_processed["cardio"].values
        
        print("Applying SMOTE for class balancing...")
        # Apply SMOTE for class imbalance
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42
        )
        
        print(f"Split data into training ({len(X_train)} samples) and testing ({len(X_test)} samples)")
        
        # Stop Spark session
        spark.stop()
        print("Spark session stopped")
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        
        # Fallback to pandas if Spark fails
        print("Falling back to pandas for preprocessing...")
        return preprocess_with_pandas(file_path)

def preprocess_with_pandas(file_path):
    """Fallback preprocessing function using pandas only."""
    # Load data
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # Prepare features - exclude id, target column, and categorical columns
    feature_cols = [col for col in df.columns if col not in ['id', 'cardio', 'bp_category', 'bp_category_encoded']]
    
    # Extract features and target
    X = df[feature_cols].values
    y = df['cardio'].values
    
    # Scale features using pandas operations
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / X_std
    
    # Apply SMOTE for class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )
    
    print(f"Processed with pandas: {len(X_train)} training samples, {len(X_test)} testing samples")
    
    return X_train, X_test, y_train, y_test