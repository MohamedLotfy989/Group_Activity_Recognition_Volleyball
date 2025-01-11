import torch
from torch import nn
from torchvision import models

class Baseline4(nn.Module):
    def __init__(self, model_path, lstm_hidden_size=512, lstm_num_layers=1, num_classes=8):
        super(Baseline4, self).__init__()
        # Load the trained ResNet model
        model = models.resnet50(pretrained=False)

        model.fc = nn.Linear(in_features=2048, out_features=8)

        # Load trained weights
        state_dict = torch.load(model_path)
        # Remove 'model.' prefix from keys
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)

        # Remove the final classification layer
        self.features = nn.Sequential(*list(model.children())[:-1])

        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad = False

        # Add LSTM layer
        self.lstm = nn.LSTM(
                            input_size=2048,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True)

        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.features(x)
        x = x.view(batch_size, seq_len, -1)  # Reshape to (batch_size, seq_len, feature_size)

        # Pass through LSTM
        lstm_out, _ = self.lstm(x)

        # Pass through fully connected layer
        fc_out = self.fc(lstm_out[:, -1, :])  # Use the last output of the LSTM

        return fc_out