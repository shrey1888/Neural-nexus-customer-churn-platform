import os
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

def tune_xgboost():
    print("Loading datasets for hyperparameter tuning...")
    train_df = pd.read_csv("data/processed/train.csv")
    train_df['Churn'] = train_df['Churn'].map({'Yes': 1, 'No': 0})
    
    X_train_raw = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn']
    
    # Align categories
    cat_cols = X_train_raw.select_dtypes(exclude='number').columns.tolist()
    X_train = pd.get_dummies(X_train_raw, columns=cat_cols, drop_first=True)
    
    # Calculate scale_pos_weight
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    
    print("Setting up grid search parameters...")
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200]
    }
    
    xgb_clf = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    print("Running GridSearchCV...")
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")
    
    # Save the best estimator
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, "Supervised_Learning/models/xgboost_tuned.pkl")
    print("Saved best tuned model to Supervised_Learning/models/xgboost_tuned.pkl")

if __name__ == "__main__":
    tune_xgboost()
