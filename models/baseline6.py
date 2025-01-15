import torch.nn as nn


class Baseline6(nn.Module):
    def __init__(self, input_size, hidden_size_frame, num_classes=8):
        super(Baseline6, self).__init__()
        # Adaptive pooling to summarize player features
        self.pool = nn.AdaptiveMaxPool1d(1)  # Pools over player dimensions
        # LSTM for frame-level temporal modeling
        self.lstm_frame = nn.LSTM(
            input_size=input_size,  # Feature size after player pooling
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

        x, _ = x[:, :, :6, :].max(axis=2)  # max over players

        # Process frames with frame-level LSTM
        _, (h_n, _) = self.lstm_frame(x)  # Output shape: (batch_size, num_frames, hidden_size_frame)

        # Use the last frame's representation for classification
        final_representation = h_n[-1]  # Shape: (batch_size, hidden_size_frame)

        # Pass through classifier to get logits
        logits = self.classifier(final_representation)  # Shape: (batch_size, num_classes)
        return logits
