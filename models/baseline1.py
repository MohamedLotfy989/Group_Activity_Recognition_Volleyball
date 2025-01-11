from torch import nn
from torchvision import models


class Baseline1(nn.Module):
    def __init__(self, num_classes=8):
        super(Baseline1, self).__init__()
        # Load pre-trained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.in_features = self.model.fc.in_features

        self.model.fc = nn.Linear(in_features=self.in_features, out_features=num_classes)

    def forward(self, x):

        return self.model(x)

