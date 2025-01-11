import torch
from torch import nn



class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 8),
        )

    def forward(self, x):
        # Max pool over players (12)
        x, _ = torch.max(x, dim=2)
        # Average pool over frames (9)
        x = torch.mean(x, dim=1)
        # Flatten to get feature vector
        x= x.view(x.size(0), -1)
        # Pass through the fully connected layer
        x = self.fc(x)
        return x