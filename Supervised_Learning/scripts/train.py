import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay)

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

def train_and_evaluate_models():
    print("Starting supervised learning model training...")
    
    # Create output directories
    os.makedirs("Supervised_Learning/models", exist_ok=True)
    os.makedirs("Supervised_Learning/outputs", exist_ok=True)
    
    # Load dataset splits
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    # Map target variable to numeric
    train_df['Churn'] = train_df['Churn'].map({'Yes': 1, 'No': 0})
    test_df['Churn'] = test_df['Churn'].map({'Yes': 1, 'No': 0})
    
    X_train_raw = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn']
    X_test_raw = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']
    
    # 1. One-hot encoding
    cat_cols = X_train_raw.select_dtypes(exclude='number').columns.tolist()
    
    # Apply get_dummies to train
    X_train = pd.get_dummies(X_train_raw, columns=cat_cols, drop_first=True)
    # Save column list for inference alignment
    model_columns = list(X_train.columns)
    joblib.dump(model_columns, "Supervised_Learning/models/model_columns.pkl")
    print(f"Saved model columns layout to models/model_columns.pkl. Total features: {len(model_columns)}")
    
    # Align test set columns to train set
    X_test = pd.get_dummies(X_test_raw, columns=cat_cols, drop_first=True)
    X_test = X_test.reindex(columns=model_columns, fill_value=0)
    
    # 2. Scale numeric columns
    num_cols = X_train_raw.select_dtypes(include='number').columns.tolist()
    # Scaler fits only on numerical columns or all columns?
    # We will fit scaler on numerical columns, but it's simpler to fit scaler on all columns of the encoded matrix
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "Supervised_Learning/models/scaler.pkl")
    print("Saved StandardScaler to models/scaler.pkl")
    
    # Handle class imbalance (churn is ~26%)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    
    # Dictionary to hold models and metrics
    fitted_models = {}
    metrics = []
    
    # 3. Model Training
    # Model A: Logistic Regression (uses scaled data)
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train_scaled, y_train)
    fitted_models['Logistic Regression'] = (lr, X_test_scaled)
    joblib.dump(lr, "Supervised_Learning/models/logistic_regression.pkl")
    
    # Model B: Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    fitted_models['Random Forest'] = (rf, X_test)
    joblib.dump(rf, "Supervised_Learning/models/random_forest.pkl")
    
    # Model C: XGBoost
    print("Training XGBoost...")
    xgb_clf = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                 scale_pos_weight=scale_pos_weight, eval_metric='logloss',
                                 random_state=42, n_jobs=-1)
    xgb_clf.fit(X_train, y_train)
    fitted_models['XGBoost'] = (xgb_clf, X_test)
    joblib.dump(xgb_clf, "Supervised_Learning/models/xgboost.pkl")
    
    # Model D: LightGBM
    print("Training LightGBM...")
    lgb_clf = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                  scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, verbose=-1)
    lgb_clf.fit(X_train, y_train)
    fitted_models['LightGBM'] = (lgb_clf, X_test)
    
    # Model E: CatBoost
    print("Training CatBoost...")
    cb_clf = cb.CatBoostClassifier(iterations=300, depth=5, learning_rate=0.05,
                                    auto_class_weights='Balanced', random_seed=42, verbose=0)
    cb_clf.fit(X_train, y_train)
    fitted_models['CatBoost'] = (cb_clf, X_test)
    
    # 4. Evaluation and Plotting
    print("Evaluating models...")
    plt.figure(figsize=(10, 8))
    
    # We will record the performance of each model
    for name, (model, test_X) in fitted_models.items():
        preds = model.predict(test_X)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(test_X)[:, 1]
        else:
            probs = preds
            
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)
        
        metrics.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC AUC': auc
        })
        
        # Add to ROC Plot
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
        
    plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 0.500)")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.savefig("Supervised_Learning/outputs/roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Save Metrics Comparison
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_excel("Supervised_Learning/outputs/model_comparison.xlsx", index=False)
    print("Model metrics saved to outputs/model_comparison.xlsx:")
    print(metrics_df.to_string(index=False))
    
    # Save confusion matrix for XGBoost (our primary model)
    xgb_model, xgb_X = fitted_models['XGBoost']
    xgb_preds = xgb_model.predict(xgb_X)
    cm = confusion_matrix(y_test, xgb_preds)
    
    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn']).plot(cmap='Blues', ax=plt.gca())
    plt.title('XGBoost Confusion Matrix')
    plt.savefig("Supervised_Learning/outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Feature Importance for XGBoost
    plt.figure(figsize=(10, 8))
    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15] # Top 15 features
    sns.barplot(x=importances[indices], y=np.array(model_columns)[indices], palette="viridis")
    plt.title("XGBoost Top 15 Feature Importances")
    plt.savefig("Supervised_Learning/outputs/feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print("Model training and evaluation successfully finished.")

if __name__ == "__main__":
    train_and_evaluate_models()
