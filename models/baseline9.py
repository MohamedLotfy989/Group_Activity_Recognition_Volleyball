import torch
from torch import nn

class Baseline9(nn.Module):
    def __init__(self):
        super(Baseline9, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 8),
        )

    def forward(self, x):
        
        # Split into teams: team_1 (players 0-5), team_2 (players 6-11)

        team_1_features, _ = x[:, :, :6, :].max(axis=2)  # max over players for team 1
        team_2_features, _ = x[:, :, 6:, :].max(axis=2)  # max over players for team 2
        clip_representation = torch.cat((team_1_features, team_2_features), dim=-1)  # Shape: (2048 + 2048)

        # Average pool over frames (9)
        clip_representation = torch.mean(clip_representation, dim=1)
        # Flatten to get feature vector
        clip_representation= clip_representation.view(clip_representation.size(0), -1)
        # Pass through the fully connected layer
        clip_representation = self.fc(clip_representation)
        return clip_representation