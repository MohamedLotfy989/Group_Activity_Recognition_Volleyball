import torch
from torch import nn
from torchvision import models


class FeatureExtractor(nn.Module):
    def __init__(self, model_path):
        super(FeatureExtractor, self).__init__()
        # Load the trained ResNet model
        model = models.resnet50(pretrained=False)

        model.fc = nn.Linear(in_features=2048, out_features=9)

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

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # Flatten to get feature vector
