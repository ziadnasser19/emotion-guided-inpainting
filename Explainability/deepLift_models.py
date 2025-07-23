import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.resnet import BasicBlock, Bottleneck
import copy


def create_deeplift_compatible_resnet(model_name="resnet18", pretrained=True, input_channels=1):
    """
    Create a DeepLift-compatible ResNet by ensuring no module reuse.
    """
    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=weights)
    else:
        raise NotImplementedError(f"Model '{model_name}' is not supported yet.")

    # Replace first conv layer for grayscale input
    model.conv1 = nn.Conv2d(
        in_channels=input_channels,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    # Fix the model for DeepLift compatibility
    fix_model_for_deeplift(model)

    return model


def fix_model_for_deeplift(model):
    """
    Comprehensive fix for DeepLift compatibility:
    1. Replace all inplace operations
    2. Ensure no module reuse
    3. Create separate ReLU instances for each use
    """
    # First pass: replace all inplace ReLUs
    replace_relu_inplace_recursive(model)

    # Second pass: fix residual blocks to avoid module reuse
    fix_residual_blocks(model)


def replace_relu_inplace_recursive(model):
    """Replace all inplace ReLU operations with non-inplace versions."""
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, nn.ReLU(inplace=False))
        else:
            replace_relu_inplace_recursive(module)


def fix_residual_blocks(model):
    """
    Fix residual blocks by creating separate ReLU instances and custom forward methods.
    """
    for module in model.modules():
        if isinstance(module, BasicBlock):
            fix_basic_block(module)
        elif isinstance(module, Bottleneck):
            fix_bottleneck_block(module)


def fix_basic_block(block):
    """Fix BasicBlock to be DeepLift compatible."""
    # Create separate ReLU instances
    block.relu1 = nn.ReLU(inplace=False)
    block.relu2 = nn.ReLU(inplace=False)

    # Store original relu for compatibility
    original_relu = block.relu

    # Define new forward method
    def deeplift_forward(self, x):
        identity = x.clone()  # Explicit clone to avoid in-place issues

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)  # Use separate relu instance

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Non-inplace addition
        out = torch.add(out, identity)  # Explicit non-inplace addition
        out = self.relu2(out)  # Use separate relu instance

        return out

    # Bind the new forward method
    block.forward = deeplift_forward.__get__(block, BasicBlock)


def fix_bottleneck_block(block):
    """Fix Bottleneck block to be DeepLift compatible."""
    # Create separate ReLU instances
    block.relu1 = nn.ReLU(inplace=False)
    block.relu2 = nn.ReLU(inplace=False)
    block.relu3 = nn.ReLU(inplace=False)
    block.relu4 = nn.ReLU(inplace=False)

    # Define new forward method
    def deeplift_forward(self, x):
        identity = x.clone()  # Explicit clone

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Non-inplace addition
        out = torch.add(out, identity)
        out = self.relu3(out)

        return out

    # Bind the new forward method
    block.forward = deeplift_forward.__get__(block, Bottleneck)


class DeepLiftCompatibleFeatureExtractor(nn.Module):
    """
    Feature extractor specifically designed for DeepLift compatibility.
    """

    def __init__(self, backbone, embedding_dim=128):
        super(DeepLiftCompatibleFeatureExtractor, self).__init__()

        # Remove classifier from backbone
        if hasattr(backbone, 'fc'):
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            in_features = backbone.fc.in_features
        elif hasattr(backbone, 'classifier'):
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            in_features = backbone.classifier[1].in_features
        else:
            raise ValueError("Unsupported backbone architecture")

        # Add global average pooling if not present
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Embedding layer
        self.embedding = nn.Linear(in_features, embedding_dim)

        # Separate ReLU for embedding if needed
        self.embedding_relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)


class DeepLiftCompatibleTripletNetwork(nn.Module):
    """
    Triplet network optimized for DeepLift compatibility.
    """

    def __init__(self, model_name="resnet18", embedding_dim=128, pretrained=True):
        super(DeepLiftCompatibleTripletNetwork, self).__init__()

        # Load DeepLift-compatible backbone
        backbone = create_deeplift_compatible_resnet(model_name, pretrained)
        self.embedding_model = DeepLiftCompatibleFeatureExtractor(backbone, embedding_dim)

    def forward(self, x):
        return self.embedding_model(x)


class DeepLiftCompatibleTripletClassifier(nn.Module):
    """
    Classifier wrapper optimized for DeepLift compatibility.
    """

    def __init__(self,
                 weights_path=None,
                 num_classes=10,
                 embedding_dim=128,
                 freeze_until=None,
                 model_name="resnet18",
                 pretrained=True,
                 device=None,
                 state_dict=None):
        super(DeepLiftCompatibleTripletClassifier, self).__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create DeepLift-compatible backbone
        if state_dict is not None:
            self.backbone = DeepLiftCompatibleTripletNetwork(
                model_name=model_name,
                embedding_dim=embedding_dim,
                pretrained=False
            )
        elif weights_path is not None:
            self.backbone = DeepLiftCompatibleTripletNetwork(
                model_name=model_name,
                embedding_dim=embedding_dim,
                pretrained=pretrained
            )
            # Load weights with proper handling
            checkpoint = torch.load(weights_path, map_location=self.device)
            self.backbone.load_state_dict(checkpoint, strict=False)
        else:
            raise ValueError("Either 'weights_path' or 'state_dict' must be provided.")

        # Layer management for progressive unfreezing
        self.feature_layers = list(self.backbone.embedding_model.features.children())

        if freeze_until is not None:
            self.freeze_layers(freeze_until)
        self.currently_frozen_until = freeze_until or 0

        # Classification head with separate components
        self.classifier = nn.Linear(embedding_dim, num_classes)

        if state_dict is not None:
            self.load_state_dict(state_dict, strict=False)

        self.to(self.device)

    def freeze_layers(self, freeze_until):
        """Freeze layers up to a certain point."""
        for i, layer in enumerate(self.feature_layers):
            requires_grad = i >= freeze_until
            for param in layer.parameters():
                param.requires_grad = requires_grad

    def progressively_unfreeze(self, step=1):
        """Progressively unfreeze layers."""
        new_freeze_until = max(0, self.currently_frozen_until - step)
        self.freeze_layers(new_freeze_until)
        self.currently_frozen_until = new_freeze_until
        print(f"[INFO] Unfroze layers from {new_freeze_until} to {len(self.feature_layers)}")

    def forward(self, x):
        """Forward pass through the network."""
        embedding = self.backbone(x)
        logits = self.classifier(embedding)
        return logits


# Updated attribution function that works with the fixed model
def attribute_image_features_deeplift(model, input_tensor, target_class, baselines=None):
    """
    Perform DeepLift attribution on the fixed model.

    Args:
        model: DeepLift-compatible model
        input_tensor: Input tensor to attribute
        target_class: Target class for attribution
        baselines: Baseline tensor (default: zero baseline)

    Returns:
        Attribution tensor
    """
    from captum.attr import DeepLift

    # Set model to evaluation mode
    model.eval()

    # Create baselines if not provided
    if baselines is None:
        baselines = torch.zeros_like(input_tensor)

    # Initialize DeepLift
    dl = DeepLift(model)

    # Clear gradients
    model.zero_grad()

    # Compute attributions
    attributions = dl.attribute(
        input_tensor,
        baselines=baselines,
        target=target_class,
        return_convergence_delta=False
    )

    return attributions


# Example usage function
def create_compatible_model_from_existing(original_weights_path, model_name="resnet18",
                                          embedding_dim=128, num_classes=10):
    """
    Create a DeepLift-compatible version of your existing model.

    Args:
        original_weights_path: Path to your original model weights
        model_name: Architecture name
        embedding_dim: Embedding dimension
        num_classes: Number of output classes

    Returns:
        DeepLift-compatible model
    """
    # Create new compatible model
    model = DeepLiftCompatibleTripletClassifier(
        weights_path=original_weights_path,
        model_name=model_name,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        pretrained=True
    )

    print("Created DeepLift-compatible model successfully!")
    print("You can now use DeepLift attribution without module reuse errors.")

    return model