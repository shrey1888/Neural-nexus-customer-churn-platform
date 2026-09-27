import os
import pandas as pd
import joblib

def load_inference_artifacts(models_dir: str = "Supervised_Learning/models"):
    model = joblib.load(os.path.join(models_dir, "xgboost.pkl"))
    columns = joblib.load(os.path.join(models_dir, "model_columns.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    return model, columns, scaler

def predict_customer_churn(customer_data: dict, models_dir: str = "Supervised_Learning/models") -> dict:
    """
    Predicts churn probability and label for a single customer.
    customer_data: dict containing feature names and values.
    """
    model, columns, scaler = load_inference_artifacts(models_dir)
    
    # Convert single dict to DataFrame
    df = pd.DataFrame([customer_data])
    
    # Apply get_dummies
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()
    df_enc = pd.get_dummies(df, columns=cat_cols)
    
    # Reindex columns to match training set
    df_aligned = df_enc.reindex(columns=columns, fill_value=0)
    
    # Predict probabilities
    prob = model.predict_proba(df_aligned)[0, 1]
    
    # Standard threshold = 0.5 (or we can tune this)
    prediction = int(prob >= 0.5)
    
    return {
        'churn_probability': float(prob),
        'churn_prediction': 'Yes' if prediction == 1 else 'No'
    }

if __name__ == "__main__":
    import json
    sample_path = "data/sample/sample_customer.json"
    if os.path.exists(sample_path):
        with open(sample_path) as f:
            customer = json.load(f)
        result = predict_customer_churn(customer)
        print("Inference Result on Sample Customer:")
        print(json.dumps(result, indent=4))
    else:
        print(f"Sample customer data not found at {sample_path}.")
