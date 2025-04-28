import torch
import torch.nn as nn

class ScoringNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=None):
        super(ScoringNetwork, self).__init__()
        if output_dim is None:
            output_dim = input_dim + 1  # Account for N agents
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)