import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

class MessageGenerationNetwork(nn.Module):
    def __init__(self, input_dim, latent_dim, message_dim, hidden_dim=128):
        super(MessageGenerationNetwork, self).__init__()
        self.latent_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )

    def forward(self, encoded_obs, prev_action):
        x = torch.cat([encoded_obs, prev_action], dim=-1)
        h = self.latent_network(x)
        mu = self.mu(h)
        log_std = self.log_std(h)
        log_std = torch.clamp(log_std, -20, 2)  
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        z = dist.rsample()  
        message = self.mlp(z)
        return message, dist