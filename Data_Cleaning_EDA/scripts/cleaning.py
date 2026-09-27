import os
import pandas as pd
import numpy as np

def clean_data(raw_path: str, output_path: str):
    """
    Loads raw Telco Churn dataset, cleans column types, handles missing values,
    drops unnecessary columns like customerID, and saves the cleaned dataset.
    """
    print(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    # Coerce TotalCharges to numeric (some values are empty spaces ' ')
    print("Coercing 'TotalCharges' to float...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan, regex=False))
    
    # Check for missing values in TotalCharges (usually where tenure is 0)
    missing_count = df['TotalCharges'].isnull().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values in 'TotalCharges'. Filling with 0.0 (tenure is 0).")
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
        
    # Drop customerID as it's not predictive
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        print("Dropped 'customerID' column.")
        
    # Clean up yes/no columns if they have trailing spaces or formatting issues (none expected in blastchar)
    # Target variable 'Churn' check
    if 'Churn' in df.columns:
        print(f"Churn value counts:\n{df['Churn'].value_counts(dropna=False)}")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path} (shape: {df.shape})")
    return df

if __name__ == "__main__":
    raw_csv = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    cleaned_csv = "data/processed/cleaned_dataset.csv"
    clean_data(raw_csv, cleaned_csv)
