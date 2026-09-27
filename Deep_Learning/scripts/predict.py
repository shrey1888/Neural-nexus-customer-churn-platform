import os
import torch
import pandas as pd
import joblib
from model import ChurnMLP

def predict_mlp_churn(customer_data: dict, models_dir: str = "Deep_Learning/models", sup_models_dir: str = "Supervised_Learning/models") -> dict:
    # Load scaling & columns layout
    columns = joblib.load(os.path.join(sup_models_dir, "model_columns.pkl"))
    scaler = joblib.load(os.path.join(sup_models_dir, "scaler.pkl"))
    
    input_dim = len(columns)
    model = ChurnMLP(input_dim)
    model.load_state_dict(torch.load(os.path.join(models_dir, "best_model.pt")))
    model.eval()
    
    # Process dictionary
    df = pd.DataFrame([customer_data])
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()
    df_enc = pd.get_dummies(df, columns=cat_cols)
    df_aligned = df_enc.reindex(columns=columns, fill_value=0)
    df_scaled = scaler.transform(df_aligned)
    
    # Feed to model
    tensor_input = torch.tensor(df_scaled, dtype=torch.float32)
    with torch.no_grad():
        logits = model(tensor_input)
        prob = torch.sigmoid(logits).item()
        
    return {
        'churn_probability': float(prob),
        'churn_prediction': 'Yes' if prob >= 0.5 else 'No'
    }

if __name__ == "__main__":
    import json
    sample_path = "data/sample/sample_customer.json"
    if os.path.exists(sample_path):
        with open(sample_path) as f:
            customer = json.load(f)
        result = predict_mlp_churn(customer)
        print("MLP Inference Result:")
        print(json.dumps(result, indent=4))
