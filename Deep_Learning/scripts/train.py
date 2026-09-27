import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import joblib

from model import ChurnMLP

class ChurnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_mlp():
    print("Starting deep learning model training...")
    os.makedirs("Deep_Learning/models", exist_ok=True)
    os.makedirs("Deep_Learning/outputs", exist_ok=True)
    
    # Load dataset splits
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/validation.csv")
    
    train_df['Churn'] = train_df['Churn'].map({'Yes': 1, 'No': 0})
    val_df['Churn'] = val_df['Churn'].map({'Yes': 1, 'No': 0})
    
    X_train_raw = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn'].values
    X_val_raw = val_df.drop(columns=['Churn'])
    y_val = val_df['Churn'].values
    
    # Load encoding schema & scaler from supervised models
    columns = joblib.load("Supervised_Learning/models/model_columns.pkl")
    scaler = joblib.load("Supervised_Learning/models/scaler.pkl")
    
    # One-hot encode and align
    cat_cols = X_train_raw.select_dtypes(exclude='number').columns.tolist()
    
    X_train_enc = pd.get_dummies(X_train_raw, columns=cat_cols)
    X_train_aligned = X_train_enc.reindex(columns=columns, fill_value=0)
    X_train_scaled = scaler.transform(X_train_aligned)
    
    X_val_enc = pd.get_dummies(X_val_raw, columns=cat_cols)
    X_val_aligned = X_val_enc.reindex(columns=columns, fill_value=0)
    X_val_scaled = scaler.transform(X_val_aligned)
    
    # Create DataLoaders
    train_dataset = ChurnDataset(X_train_scaled, y_train)
    val_dataset = ChurnDataset(X_val_scaled, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    input_dim = X_train_scaled.shape[1]
    model = ChurnMLP(input_dim)
    
    # Class weights for imbalance
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    
    epochs = 100
    patience = 10
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        correct_train = 0
        total_train = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
            
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        train_acc = correct_train / total_train
        
        # Validation Loop
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_acc = correct_val / total_val
        
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print logs
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
            
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "Deep_Learning/models/best_model.pt")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break
                
    # Save Loss and Accuracy Curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PyTorch MLP Training Loss Curve")
    plt.legend()
    plt.savefig("Deep_Learning/outputs/loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("PyTorch MLP Training Accuracy Curve")
    plt.legend()
    plt.savefig("Deep_Learning/outputs/accuracy_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print("Deep learning model training successfully finished.")

if __name__ == "__main__":
    train_mlp()
