import torch
import torch.nn as nn



class Baseline8(nn.Module):
    def __init__(self, input_size, hidden_size_player, hidden_size_frame, num_classes=8):
        super(Baseline8, self).__init__()
        # LSTM for player-level temporal modeling
        self.player_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_player,
            num_layers=1,
            batch_first=True
        )
        # Adaptive pooling to summarize each team's features
        self.team_pool = nn.AdaptiveMaxPool1d(1)
        # LSTM for scene-level temporal modeling
        self.scene_lstm = nn.LSTM(
            input_size=hidden_size_player * 2,  # Combined team features
            hidden_size=hidden_size_frame,
            num_layers=1,
            batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(2048)
        self.layer_norm2 = nn.LayerNorm(1024)
        # Classifier for final prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size_frame, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch_size, num_frames, num_players, feature_dim)
        batch_size, num_frames, num_players, feature_dim = x.size()

        # Reshape for player-level LSTM: merge batch and frame dimensions
        x = x.view(batch_size * num_frames, num_players, feature_dim)
        x = self.layer_norm1(x)
        # Process each player with LSTM
        x, _ = self.player_lstm(x)  # Output shape: (batch_size * num_frames, num_players, hidden_size_player)

        # Split features into two teams (first 6 and last 6 players)
        team_1 = x[:, :6, :]  # Shape: (batch_size * num_frames, 6, hidden_size_player)
        team_2 = x[:, 6:, :]  # Shape: (batch_size * num_frames, 6, hidden_size_player)

        # Pool across players for each team
        team_1 = self.team_pool(team_1.permute(0, 2, 1)).squeeze(-1)  # Shape: (batch_size * num_frames, hidden_size_player)
        team_2 = self.team_pool(team_2.permute(0, 2, 1)).squeeze(-1)  # Shape: (batch_size * num_frames, hidden_size_player)

        # Concatenate team features
        scene_representation = torch.cat([team_1, team_2], dim=1)  # Shape: (batch_size * num_frames, hidden_size_player * 2)

        # Reshape back to batch and frame dimensions
        scene_representation = scene_representation.view(batch_size, num_frames, -1)
        scene_representation = self.layer_norm1(scene_representation)
        # Process scene features with scene-level LSTM
        scene_output, _ = self.scene_lstm(scene_representation)  # Shape: (batch_size, num_frames, hidden_size_frame)

        # Use the last frame's representation for classification
        final_representation = scene_output[:, -1, :]  # Shape: (batch_size, hidden_size_frame)

        # Pass through classifier to get logits
        logits = self.classifier(final_representation)  # Shape: (batch_size, num_classes)
        return logits

