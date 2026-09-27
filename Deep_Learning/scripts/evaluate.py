import os
import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, roc_auc_score
from model import ChurnMLP

def evaluate_mlp():
    print("Evaluating PyTorch MLP model on test set...")
    os.makedirs("Deep_Learning/outputs", exist_ok=True)
    
    # Load dataset
    test_df = pd.read_csv("data/processed/test.csv")
    test_df['Churn'] = test_df['Churn'].map({'Yes': 1, 'No': 0})
    
    X_test_raw = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn'].values
    
    # Load artifacts
    columns = joblib.load("Supervised_Learning/models/model_columns.pkl")
    scaler = joblib.load("Supervised_Learning/models/scaler.pkl")
    
    # Load model
    input_dim = len(columns)
    model = ChurnMLP(input_dim)
    model.load_state_dict(torch.load("Deep_Learning/models/best_model.pt"))
    model.eval()
    
    # One-hot encode & scale
    cat_cols = X_test_raw.select_dtypes(exclude='number').columns.tolist()
    X_test_enc = pd.get_dummies(X_test_raw, columns=cat_cols)
    X_test_aligned = X_test_enc.reindex(columns=columns, fill_value=0)
    X_test_scaled = scaler.transform(X_test_aligned)
    
    # Predict
    tensor_input = torch.tensor(X_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        logits = model(tensor_input)
        probs = torch.sigmoid(logits).numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        
    report = classification_report(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    report_text = f"PyTorch MLP Model Test Evaluation\n"
    report_text += f"===============================\n"
    report_text += f"ROC AUC Score: {auc:.4f}\n\n"
    report_text += report
    
    print("\nTest Evaluation Summary:")
    print(report_text)
    
    with open("Deep_Learning/outputs/classification_report.txt", "w") as f:
        f.write(report_text)
        
    print("Saved report to Deep_Learning/outputs/classification_report.txt")

if __name__ == "__main__":
    evaluate_mlp()
