import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class EmotionDetector(nn.Module):
    def __init__(self, num_classes=8, dropout_p=0.5):
        super(EmotionDetectorResNet18, self).__init__()

        # Load pretrained ResNet-18
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Modify first conv layer for 1-channel grayscale input
        old_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        with torch.no_grad():
            self.model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        # Replace the classifier with dropout + final FC
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)