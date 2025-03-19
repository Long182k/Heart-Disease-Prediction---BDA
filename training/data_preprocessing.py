import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Advanced Feature Engineering
    df['age_years'] = df['age'] // 365
    df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
    df['bp_category_encoded'] = df['bp_category'].astype('category').cat.codes
    
    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)  # Specify numeric_only=True
    
    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = pd.Categorical(df[col]).codes
    
    # Split features and target
    X = df.drop(['id', 'cardio'], axis=1)
    y = df['cardio']
    
    # Apply SMOTE for class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    
    return train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)