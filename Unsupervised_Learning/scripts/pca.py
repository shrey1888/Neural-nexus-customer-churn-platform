import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

def run_pca_visualization():
    print("Running PCA dimension reduction...")
    os.makedirs("Unsupervised_Learning/outputs", exist_ok=True)
    
    df = pd.read_csv("data/processed/cleaned_dataset.csv")
    
    if 'Cluster' not in df.columns:
        raise ValueError("Run clustering.py first to append Cluster labels.")
        
    clusters = df['Cluster']
    X_raw = df.drop(columns=['Churn', 'Cluster'], errors='ignore')
    
    # One-hot encode
    cat_cols = X_raw.select_dtypes(exclude='number').columns.tolist()
    X_enc = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enc)
    
    # PCA to 2 components
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
    pca_df['Cluster'] = clusters
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='PCA1', y='PCA2', 
        hue='Cluster', 
        palette='Set1', 
        data=pca_df, 
        alpha=0.6,
        edgecolor=None
    )
    plt.title('Customer Clusters Visualized in 2D (PCA)', fontsize=14)
    plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.legend(title='Cluster')
    
    plt.savefig("Unsupervised_Learning/outputs/pca_visualization.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # Scatter plot of tenure and MonthlyCharges colored by clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='tenure', y='MonthlyCharges',
        hue='Cluster', palette='Set1',
        data=df, alpha=0.5
    )
    plt.title('Customer Segments: Tenure vs Monthly Charges', fontsize=14)
    plt.savefig("Unsupervised_Learning/outputs/clusters.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print("PCA and cluster visualizations saved successfully.")

if __name__ == "__main__":
    run_pca_visualization()
