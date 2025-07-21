import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross Entropy Loss with Label Smoothing regularization."""
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, outputs, targets):
        log_probs = F.log_softmax(outputs, dim=1)
        num_classes = outputs.size(1)
        
        # Create smoothed targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


class WeightedLabelSmoothingCE(nn.Module):
    """Weighted Cross Entropy Loss with Label Smoothing."""
    
    def __init__(self, class_weights, smoothing=0.1, device='cuda'):
        super(WeightedLabelSmoothingCE, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.class_weights = class_weights.to(device)
        self.device = device

    def forward(self, outputs, targets):
        log_probs = F.log_softmax(outputs, dim=1)
        num_classes = outputs.size(1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), self.confidence)
        
        # Apply class weights
        weights = self.class_weights[targets]
        weighted_loss = -torch.sum(true_dist * log_probs, dim=1) * weights
        
        return torch.mean(weighted_loss)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha=1.0, gamma=2.0, smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, outputs, targets):
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Add label smoothing if specified
        if self.smoothing > 0:
            log_probs = F.log_softmax(outputs, dim=1)
            num_classes = outputs.size(1)
            
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(self.smoothing / (num_classes - 1))
                true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.smoothing)
            
            smooth_loss = torch.mean(torch.sum(-true_dist * log_probs, dim=1))
            focal_loss = focal_loss.mean() * 0.7 + smooth_loss * 0.3
        else:
            focal_loss = focal_loss.mean()
            
        return focal_loss


class MixUpLoss(nn.Module):
    """Enhanced MixUp Loss with label smoothing support."""
    
    def __init__(self, alpha=0.2, smoothing=0.1):
        super(MixUpLoss, self).__init__()
        self.alpha = alpha
        self.smoothing = smoothing
        if smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets, targets_b=None, lam=1.0):
        if targets_b is None or lam == 1.0:
            return self.criterion(outputs, targets)
        
        # MixUp: combine two targets
        return lam * self.criterion(outputs, targets) + (1 - lam) * self.criterion(outputs, targets_b)


def get_criterion(config, class_weights=None, device='cuda'):
    """Factory function to create appropriate loss function."""
    
    smoothing = getattr(config, 'LABEL_SMOOTHING', 0.0)
    criterion_type = config.CRITERION
    
    if criterion_type == "cross_entropy":
        if smoothing > 0:
            return LabelSmoothingCrossEntropy(smoothing=smoothing)
        else:
            return nn.CrossEntropyLoss()
    
    elif criterion_type == "weighted_loss":
        if class_weights is None:
            raise ValueError("Class weights required for weighted loss")
        if smoothing > 0:
            return WeightedLabelSmoothingCE(class_weights, smoothing=smoothing, device=device)
        else:
            return nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    elif criterion_type == "focal_loss":
        return FocalLoss(alpha=1.0, gamma=2.0, smoothing=smoothing)
    
    elif criterion_type == "mixup":
        return MixUpLoss(alpha=0.2, smoothing=smoothing)
    
    else:
        raise ValueError(f"Unsupported criterion: {criterion_type}")
