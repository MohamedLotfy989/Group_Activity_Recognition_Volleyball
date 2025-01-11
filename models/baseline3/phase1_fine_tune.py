from torch import nn
from torchvision import models


class Baseline3(nn.Module):
    def __init__(self, num_classes=9):
        super(Baseline3, self).__init__()
        # Load pre-trained ResNet50
        self.model = models.resnet50(pretrained=True)
        self.in_features = self.model.fc.in_features

        self.model.fc = nn.Linear(in_features=self.in_features, out_features=num_classes)

    def forward(self, x):

        return self.model(x)

