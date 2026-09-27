import os
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import joblib

def evaluate_clustering():
    print("Evaluating customer clusters...")
    df = pd.read_csv("data/processed/cleaned_dataset.csv")
    
    if 'Cluster' not in df.columns:
        raise ValueError("Run clustering.py first.")
        
    clusters = df['Cluster']
    X_raw = df.drop(columns=['Churn', 'Cluster'], errors='ignore')
    
    # One-hot encode
    cat_cols = X_raw.select_dtypes(exclude='number').columns.tolist()
    X_enc = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enc)
    
    # Compute metrics
    sil = silhouette_score(X_scaled, clusters)
    db = davies_bouldin_score(X_scaled, clusters)
    ch = calinski_harabasz_score(X_scaled, clusters)
    
    print("\nClustering Validation Metrics:")
    print(f"  Silhouette Score (higher is better):          {sil:.4f}")
    print(f"  Davies-Bouldin Index (lower is better):        {db:.4f}")
    print(f"  Calinski-Harabasz Index (higher is better):    {ch:.2f}")

if __name__ == "__main__":
    evaluate_clustering()
