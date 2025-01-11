import torch.nn as nn



class Baseline7(nn.Module):
    def __init__(self, input_size, hidden_size_player, hidden_size_frame, num_classes=8):
        super(Baseline7, self).__init__()
        # LSTM for player-level temporal modeling
        self.player_lstm = nn.LSTM(
            input_size=input_size,  # Input feature size per player
            hidden_size=hidden_size_player,  # Hidden size for LSTM
            num_layers=1,  # Number of LSTM layers
            batch_first=True  # Input batch comes first
        )
        # Adaptive pooling to summarize player features
        self.pool = nn.AdaptiveMaxPool1d(1)  # Pools over player dimensions
        # LSTM for frame-level temporal modeling
        self.lstm_frame = nn.LSTM(
            input_size=hidden_size_player,  # Feature size after player pooling
            hidden_size=hidden_size_frame,  # Hidden size for frame LSTM
            num_layers=1,  # Number of LSTM layers
            batch_first=True  # Input batch comes first
        )
        # Classifier for final prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size_frame, 512),  # First dense layer
            nn.BatchNorm1d(512),  # Batch normalization
            nn.ReLU(),  # Activation function
            nn.Dropout(0.5),  # Regularization with dropout
            nn.Linear(512, num_classes)  # Final dense layer for predictions
        )

    def forward(self, x):
        # Input shape: (batch_size, num_frames, num_players, feature_dim)
        batch_size, num_frames, num_players, feature_dim = x.size()

        # Reshape input for player-level LSTM: merge batch and player dimensions
        x = x.view(batch_size * num_players, num_frames, feature_dim)

        # Process each player's sequence with LSTM
        player_output, _ = self.player_lstm(x)  # LSTM output shape: (batch_size * num_players, num_frames, hidden_size_player)

        # Reshape to include the player dimension for all frames
        player_output = player_output.view(batch_size, num_players, num_frames,
                                           -1)  # (batch_size, num_players, num_frames, hidden_dim_player)

        player_output = player_output.permute(0, 2, 1, 3)  # (batch_size, num_frames, num_players, hidden_dim_player)

        player_output = player_output.contiguous()
        # Pool player features to a single representation per frame
        player_output_flat = player_output.view(batch_size * num_frames, num_players, -1)
        frame_rep = self.pool(player_output_flat.permute(0, 2, 1)).squeeze(-1)  # Shape: (batch_size * num_frames, hidden_size_player)
        # frame_rep = frame_rep.contiguous()

        # Reshape for frame-level LSTM
        frame_rep = frame_rep.view(batch_size, num_frames, -1)  # Shape: (batch_size, num_frames, hidden_size_player)

        # Process frames with frame-level LSTM
        frame_out, _ = self.lstm_frame(frame_rep)  # Output shape: (batch_size, num_frames, hidden_size_frame)

        # Use the last frame's representation for classification
        final_representation = frame_out[:, -1, :]  # Shape: (batch_size, hidden_size_frame)

        # Pass through classifier to get logits
        logits = self.classifier(final_representation)  # Shape: (batch_size, num_classes)
        return logits

