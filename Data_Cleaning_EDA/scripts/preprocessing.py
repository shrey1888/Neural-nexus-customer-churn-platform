import os
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_and_split(cleaned_path: str, output_dir: str, random_state: int = 42):
    """
    Reads cleaned dataset and performs a stratified split into train, validation, and test sets,
    preserving categorical variables for down-stream custom encoders.
    """
    print(f"Reading cleaned dataset from {cleaned_path}...")
    df = pd.read_csv(cleaned_path)
    
    if 'Churn' not in df.columns:
        raise ValueError("Target variable 'Churn' not found in dataset columns.")
        
    # Split into train (70%) and temporary (30%)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.30, 
        stratify=df['Churn'], 
        random_state=random_state
    )
    
    # Split temporary into validation (50% of temp = 15% of total) and test (50% of temp = 15% of total)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        stratify=temp_df['Churn'], 
        random_state=random_state
    )
    
    print(f"Dataset split results:")
    print(f"  Train shape: {train_df.shape} | Churn Rate: {train_df['Churn'].value_counts(normalize=True).get('Yes', 0.0):.2%}")
    print(f"  Val shape:   {val_df.shape} | Churn Rate: {val_df['Churn'].value_counts(normalize=True).get('Yes', 0.0):.2%}")
    print(f"  Test shape:  {test_df.shape} | Churn Rate: {test_df['Churn'].value_counts(normalize=True).get('Yes', 0.0):.2%}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "validation.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved splits to:\n  - {train_path}\n  - {val_path}\n  - {test_path}")

if __name__ == "__main__":
    cleaned_csv = "data/processed/cleaned_dataset.csv"
    output_directory = "data/processed"
    if os.path.exists(cleaned_csv):
        preprocess_and_split(cleaned_csv, output_directory)
    else:
        print(f"Error: {cleaned_csv} not found. Run cleaning.py and feature_engineering.py first.")
