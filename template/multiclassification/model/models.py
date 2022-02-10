import timm
import torch
import torch.nn as nn

class Custom_EfficientNet(nn.Module):
    def __init__(self, config: dict):
        super(Custom_EfficientNet, self).__init__()
        self.net = timm.create_model(config.model.name, pretrained=True)
        in_features = self.net.classifier.in_features
        self.net.classifier = nn.Linear(in_features, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
