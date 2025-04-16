import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('/kaggle/input/cardiovascular-disease/cardio_data_processed.csv')

# First split: train (80%) and temp (20%)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['cardio'])

# Second split: validation (10%) and test (10%) from temp
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['cardio'])

# Now you have train_df, val_df, and test_df
print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")

# Pass to your workflow
from main import run_sklearn_xgb_workflow
import joblib

model, results = run_sklearn_xgb_workflow(train_df, val_df, test_df,'/kaggle/working/')

# Save the trained model as a .pkl file
joblib.dump(model, '/kaggle/working/xgb_model.pkl')
print("Model saved to /kaggle/working/xgb_model.pkl")