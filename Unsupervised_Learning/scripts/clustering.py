import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib

def perform_clustering():
    print("Starting unsupervised learning clustering...")
    os.makedirs("Unsupervised_Learning/models", exist_ok=True)
    os.makedirs("Unsupervised_Learning/outputs", exist_ok=True)
    
    df = pd.read_csv("data/processed/cleaned_dataset.csv")
    
    # Drop target column for clustering
    X_raw = df.drop(columns=['Churn'], errors='ignore')
    
    # One-hot encode categorical features
    cat_cols = X_raw.select_dtypes(exclude='number').columns.tolist()
    X_enc = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enc)
    
    # Save the unsupervised scaler
    joblib.dump(scaler, "Unsupervised_Learning/models/scaler.pkl")
    joblib.dump(list(X_enc.columns), "Unsupervised_Learning/models/clustering_columns.pkl")
    print("Saved unsupervised scaler and columns list.")
    
    # 1. Generate Elbow Curve
    print("Generating Elbow Curve...")
    inertias = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, 'bo-', markersize=6)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.savefig("Unsupervised_Learning/outputs/elbow_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 2. Compute Silhouette Scores
    print("Computing Silhouette Scores...")
    sil_scores = []
    k_sil = range(2, 6) # Silhouette not defined for k=1
    for k in k_sil:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, cluster_labels)
        sil_scores.append(score)
        print(f"  k={k} | Silhouette Score: {score:.4f}")
        
    plt.figure(figsize=(8, 5))
    plt.bar(k_sil, sil_scores, color='skyblue')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Comparison')
    plt.savefig("Unsupervised_Learning/outputs/silhouette_score.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # We choose k=4 clusters for segmentation
    optimal_k = 4
    print(f"Fitting optimal K-Means model with k={optimal_k}...")
    kmeans_opt = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans_opt.fit_predict(X_scaled)
    
    joblib.dump(kmeans_opt, "Unsupervised_Learning/models/kmeans.pkl")
    print("Saved optimal K-Means model to models/kmeans.pkl")
    
    # Add clusters back to df
    df['Cluster'] = clusters
    df.to_csv("data/processed/cleaned_dataset.csv", index=False)
    print("Added cluster labels back to cleaned_dataset.csv.")
    
    # 3. Analyze Clusters
    cluster_profiles = df.groupby('Cluster')[['tenure', 'MonthlyCharges', 'TotalCharges', 'ServiceCount']].mean()
    cluster_churn = df.groupby('Cluster')['Churn'].value_counts(normalize=True).unstack().get('Yes', 0.0)
    cluster_profiles['ChurnRate'] = cluster_churn
    cluster_profiles['Size'] = df['Cluster'].value_counts()
    
    print("\nCluster Profiles Summary:")
    print(cluster_profiles)
    
    return X_scaled, clusters

if __name__ == "__main__":
    perform_clustering()
