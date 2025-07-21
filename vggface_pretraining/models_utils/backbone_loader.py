import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights

def load_backbone(model_name="resnet18", pretrained=True):
    """
    Dynamically load backbone architecture with optional pre-trained weights.

    Args:
        model_name (str): Backbone architecture (e.g., 'resnet18').
        pretrained (bool): Whether to load ImageNet-pretrained weights.

    Returns:
        nn.Module: Backbone model
    """
    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)

        # Replace first conv layer for grayscale (1 channel)
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        return model
    elif model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=weights)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    else:
        raise NotImplementedError(f"Backbone '{model_name}' is not supported yet.")
