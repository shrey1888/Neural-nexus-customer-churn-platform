import os
import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering to the telecom churn dataset.
    """
    df = df.copy()
    
    # 1. Tenure groupings
    # Bins: 0-12, 12-24, 24-48, 48-60, 60+
    print("Creating tenure groups...")
    df['TenureGroup'] = pd.cut(
        df['tenure'], 
        bins=[-1, 12, 24, 48, 60, np.inf], 
        labels=['0-12m', '12-24m', '24-48m', '48-60m', '60m+']
    ).astype(str)
    
    # 2. Service Count: count the number of services subscribed to
    print("Calculating service counts...")
    services = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 
        'StreamingTV', 'StreamingMovies'
    ]
    
    # We count 'Yes' values for each service column
    service_count = np.zeros(len(df))
    for col in services:
        if col in df.columns:
            service_count += (df[col] == 'Yes').astype(int)
            
    # Add InternetService if it is not 'No'
    if 'InternetService' in df.columns:
        service_count += (df['InternetService'] != 'No').astype(int)
        
    df['ServiceCount'] = service_count
    
    # 3. Monthly charges to total charges ratio (approximates months billed vs actual tenure)
    print("Calculating charges ratio...")
    df['ChargesRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1e-5)
    
    # 4. Has fiber optic internet (highly correlated with churn in Telco)
    if 'InternetService' in df.columns:
        df['HasFiberOptic'] = (df['InternetService'] == 'Fiber optic').astype(int)
    
    # 5. Contract type indicators (month-to-month has much higher churn)
    if 'Contract' in df.columns:
        df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
        
    # 6. Automatic payment flag
    if 'PaymentMethod' in df.columns:
        df['IsAutomaticPayment'] = df['PaymentMethod'].str.contains('automatic', case=False).astype(int)
        
    print(f"Feature engineering complete. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    cleaned_path = "data/processed/cleaned_dataset.csv"
    if os.path.exists(cleaned_path):
        df = pd.read_csv(cleaned_path)
        df_feats = add_features(df)
        # overwrite or save as feature engineered csv
        df_feats.to_csv(cleaned_path, index=False)
        print(f"Saved feature engineered data back to {cleaned_path}")
    else:
        print(f"Error: {cleaned_path} does not exist. Run cleaning.py first.")
