import torch
import torch.nn as nn

class ChurnMLP(nn.Module):
    def __init__(self, input_dim: int):
        super(ChurnMLP, self).__init__()
        
        self.network = nn.Sequential(
            # Hidden layer 1
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layer 2
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layer 3
            nn.Linear(32, 16),
            nn.ReLU(),
            
            # Output layer (returns logits)
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.network(x)
