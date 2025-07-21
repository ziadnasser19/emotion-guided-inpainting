import torch
import torch.nn as nn
import torch.nn.functional as F

from vggface_pretraining.models_utils.backbone_loader import load_backbone
from vggface_pretraining.models_utils.triplet_network_loader import load_triplet_model


class FeatureExtractor(nn.Module):
    """
    Wraps any backbone model and adapts it to output a normalized embedding.
    """
    def __init__(self, backbone, embedding_dim=128):
        """
        Args:
            backbone (nn.Module): Backbone model with a feature extractor.
            embedding_dim (int): Size of the output embedding.
        """
        super(FeatureExtractor, self).__init__()

        # Remove classifier from backbone (assumes standard torchvision format)
        if hasattr(backbone, 'fc'):  # For ResNet-type models
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            in_features = backbone.fc.in_features
        elif hasattr(backbone, 'classifier'):  # For MobileNet, etc.
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            in_features = backbone.classifier[1].in_features
        else:
            raise ValueError("Unsupported backbone architecture")

        # New embedding layer
        self.embedding = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        """
        Forward pass to extract and normalize embeddings.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalization


class TripletNetwork(nn.Module):
    """
    Triplet network wrapper that outputs embeddings from a backbone model.
    """
    def __init__(self, model_name="resnet18", embedding_dim=128, pretrained=True):
        """
        Args:
            model_name (str): Name of the backbone model.
            embedding_dim (int): Size of the output feature embedding.
            pretrained (bool): Use pretrained weights or not.
        """
        super(TripletNetwork, self).__init__()

        # Load the backbone and wrap it in a feature extractor
        backbone = load_backbone(model_name, pretrained)
        self.embedding_model = FeatureExtractor(backbone, embedding_dim)

    def forward(self, x):
        """
        Forward pass through the embedding network.
        """
        return self.embedding_model(x)

# ---- Classifier Wrapper ----

class TripletClassifier(nn.Module):
    def __init__(self,
                 weights_path=None,
                 num_classes=10,
                 embedding_dim=128,
                 freeze_until=None,
                 model_name="resnet18",
                 pretrained=True,
                 device=None,
                 state_dict=None):
        super(TripletClassifier, self).__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if state_dict is not None:
            self.backbone = TripletNetwork(model_name=model_name, embedding_dim=embedding_dim, pretrained=False)
        elif weights_path is not None:
            self.backbone = load_triplet_model(
                weights_path=weights_path,
                model_name=model_name,
                embedding_dim=embedding_dim,
                pretrained=pretrained,
                device=self.device
            )
        else:
            raise ValueError("Either 'weights_path' or 'state_dict' must be provided.")

        # Expose layers for progressive unfreezing
        self.feature_layers = list(self.backbone.embedding_model.features.children())

        # Freeze layers initially
        if freeze_until is not None:
            self.freeze_layers(freeze_until)
        self.currently_frozen_until = freeze_until or 0

        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)

        if state_dict is not None:
            self.load_state_dict(state_dict)

        self.to(self.device)

    def freeze_layers(self, freeze_until):
        for i, layer in enumerate(self.feature_layers):
            requires_grad = i >= freeze_until
            for param in layer.parameters():
                param.requires_grad = requires_grad

    def progressively_unfreeze(self, step=1):
        new_freeze_until = max(0, self.currently_frozen_until - step)
        self.freeze_layers(new_freeze_until)
        self.currently_frozen_until = new_freeze_until
        print(f"[INFO] Unfroze layers from {new_freeze_until} to {len(self.feature_layers)}")

    def forward(self, x):
        embedding = self.backbone(x)
        logits = self.classifier(embedding)
        return logits