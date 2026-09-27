import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def evaluate_saved_model(model_name: str = "xgboost"):
    print(f"Evaluating {model_name} model on test set...")
    
    # Load dataset
    test_df = pd.read_csv("data/processed/test.csv")
    test_df['Churn'] = test_df['Churn'].map({'Yes': 1, 'No': 0})
    
    X_test_raw = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']
    
    # Load artifacts
    models_dir = "Supervised_Learning/models"
    model = joblib.load(os.path.join(models_dir, f"{model_name}.pkl"))
    columns = joblib.load(os.path.join(models_dir, "model_columns.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    
    # One-hot encode and align
    cat_cols = X_test_raw.select_dtypes(exclude='number').columns.tolist()
    X_test_enc = pd.get_dummies(X_test_raw, columns=cat_cols)
    X_test_aligned = X_test_enc.reindex(columns=columns, fill_value=0)
    
    # If logistic regression, use scaled data
    if model_name == "logistic_regression":
        X_eval = scaler.transform(X_test_aligned)
    else:
        X_eval = X_test_aligned
        
    # Get predictions
    preds = model.predict(X_eval)
    probs = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else preds
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    print(f"ROC AUC Score: {roc_auc_score(y_test, probs):.4f}")

if __name__ == "__main__":
    evaluate_saved_model("xgboost")
