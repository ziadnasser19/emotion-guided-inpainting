import torch
import torch.nn as nn
from torchvision import models


class EmotionDetector(nn.Module):

#constructor
    def __init__(self, num_classes, fine_tune="full"):
        """
        Initializes the EmotionDetector model using ResNet50.
        Parameters:
        - num_classes (int): Number of emotion classes to predict.
        - fine_tune options:
            "none": freeze all pretrained layers
            "partial": unfreeze last block (layer4)
            "full": train all layers

        """
        super(EmotionDetector, self).__init__()

        # Load pre-trained ResNet50 model
        self.base_model = models.resnet50(pretrained=True)

        if fine_tune == "none":
            for param in self.base_model.parameters():
                param.requires_grad = False

        elif fine_tune == "partial":
            # Freeze all first
            for param in self.base_model.parameters():
                param.requires_grad = False
            # Then unfreeze layer4 (final block)
            for param in self.base_model.layer4.parameters():
                param.requires_grad = True

        # If "full", no need to freeze anything

        # Replace the final fully connected layer
        in_features = self.base_model.fc.in_features  # Size of the last hidden layer
        self.base_model.fc = nn.Linear(in_features, num_classes)  # New output layer
#forward
    def forward(self, x):
        """
        Defines the forward pass of the model.
        Parameters:
        - x (Tensor): Input image batch.
        Returns:
        - logits (Tensor): Raw scores for each emotion class.
        """
        return self.base_model(x)