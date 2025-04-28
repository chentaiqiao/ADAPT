import torch.nn as nn

class ObservationReconstructionNetwork(nn.Module):
    def __init__(self, message_dim, obs_dim, hidden_dim=128):
        super(ObservationReconstructionNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )

    def forward(self, messages):
        recon_obs = self.network(messages)  # [batch_size, obs_dim]
        return recon_obs