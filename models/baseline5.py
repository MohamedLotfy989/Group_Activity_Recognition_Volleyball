import torch
import torch.nn as nn


class Baseline5(nn.Module):
    def __init__(self, input_size, hidden_size_player, num_classes=8):
        super(Baseline5, self).__init__()
        # LSTM layer for processing player-level temporal features
        self.player_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_player,
            num_layers=1,
            batch_first=True
        )
        # Fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size_player, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, num_frames, num_players, feature_dim)
        batch_size, num_frames, num_players, feature_dim = x.size()

        # Reshape input for LSTM processing
        x = x.view(batch_size * num_players, num_frames, feature_dim)

        # Pass through LSTM and take the last hidden state
        _, (h_n, _) = self.player_lstm(x)
        x = h_n[-1]  # Shape: (batch_size * num_players, hidden_size_player)

        # Reshape back to batch size with player-level representations
        x = x.view(batch_size, num_players, -1)  # Shape: (batch_size, num_players, hidden_size_player)

        # Pool player representations to a single vector per batch
        pooled_features = torch.max(x, dim=1)[0]  # Pooling over players

        # Pass pooled features through classifier
        logits = self.classifier(pooled_features)
        return logits


