import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
from enum import Enum
from typing import List, Optional, Tuple

class LandmassLoss(nn.Module):
    """
    A Loss function to encourage the model to accurately predict the overall mangrove coverage in the image.

    This will allow the model to train on the overall size of the mangrove area, rather than individual pixels.
    It is also a very simple calculation - essentially the normalized absolute difference.
    """
    def __init__(self):
        super(LandmassLoss, self).__init__()
    def forward(self, prediction, target):
        return (prediction.sum() - target.sum()) / (target.sum() + 1)
class JaccardLoss(nn.Module):
    """
    Jaccard Loss (IoU Loss) for segmentation
    
    This loss directly optimizes the Intersection over Union metric
    """
    def __init__(self, num_classes=1, ignore_index=255, smooth=1e-6, weight=None):
        """
        Args:
            num_classes: Number of segmentation classes
            ignore_index: Index to ignore in loss calculation (default: 255)
            smooth: Smoothing factor to avoid division by zero
            weight: Optional class weights for handling imbalance
        """
        super(JaccardLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.weight = weight
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, C, H, W] - model logits
            targets: [B, H, W] or [B, 1, H, W] - ground truth class indices
        """
        
        # Normalize target shape first
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
        
        if self.num_classes > 1:
            # Multi-class case
            predictions = F.softmax(predictions, dim=1)  # [B, C, H, W]
            targets_onehot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()  # [B, C, H, W]
            
            # Handle ignore_index
            if self.ignore_index is not None:
                mask = (targets != self.ignore_index).float().unsqueeze(1)  # [B, 1, H, W]
                predictions = predictions * mask
                targets_onehot = targets_onehot * mask
            
            # Calculate intersection and union
            intersection = (predictions * targets_onehot).sum(dim=(2, 3))  # [B, C]
            union = predictions.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3)) - intersection
            
        else:
            # Binary case
            if predictions.shape[1] == 2:
                predictions = F.softmax(predictions, dim=1)[:, 1]  # [B, H, W]
            else:
                predictions = torch.sigmoid(predictions.squeeze(1))  # [B, H, W]
            
            # Ensure targets is float and same shape as predictions [B, H, W]
            targets = targets.float()
            
            # Handle ignore_index
            if self.ignore_index is not None:
                mask = (targets != self.ignore_index).float()  # [B, H, W]
                predictions = predictions * mask
                targets = targets * mask
            
            # Add channel dimension for consistent operations
            predictions = predictions.unsqueeze(1)  # [B, 1, H, W]
            targets = targets.unsqueeze(1)  # [B, 1, H, W]
            
            # Calculate intersection and union
            intersection = (predictions * targets).sum(dim=(2, 3))  # [B, 1]
            union = predictions.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection  # [B, 1]
        
        # Calculate IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Apply class weights if provided
        if self.weight is not None:
            iou = iou * self.weight.to(iou.device)
        
        # Return 1 - mean IoU as loss
        return 1 - iou.mean()
    
class FocalJaccardLoss(nn.Module):
    """
    Focal Jaccard Loss - focuses on hard examples by applying a focal term to the Jaccard loss
    """
    def __init__(self, num_classes=1, ignore_index=255, smooth=1e-6):
        super(FocalJaccardLoss, self).__init__()
        self.jaccard_loss = JaccardLoss(num_classes, ignore_index, smooth)
        self.focal = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75, smooth=smooth)
        
    def forward(self, predictions, targets):
        jaccard = self.jaccard_loss(predictions, targets)
        focal = self.focal(predictions, targets)
        focal_jaccard = 0.6 * jaccard + 0.4 * focal
        return focal_jaccard
    
class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss - excellent for punishing false negatives
    alpha controls FP vs FN trade-off
    gamma controls focus on hard examples
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # Weight for FN (higher = punish FN more)
        self.beta = beta    # Weight for FP (should be 1-alpha)
        self.gamma = gamma  # Focal parameter
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        
        # Get probabilities
        if predictions.shape[1] == 2:
            preds = F.softmax(predictions, dim=1)[:, 1]
        else:
            preds = torch.sigmoid(predictions.squeeze(1))
        
        targets = targets.float()
        
        # Flatten
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # Calculate Tversky components
        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()
        
        # Tversky Index
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        # Focal Tversky Loss
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg

    def forward(self, fake, real):
        f_fake = self.vgg(fake)
        f_real = self.vgg(real)
        return F.l1_loss(f_fake, f_real)

class _ConvNextType(Enum):
    """Available ConvNext model types"""
    TINY = "tiny"
    SMALL = "small"
    BASE = "base"
    LARGE = "large"

class ConvNextPerceptualLoss(nn.Module):
    """
    Taken from https://github.com/sypsyp97/convnext_perceptual_loss/tree/main
    Perceptual loss using pretrained ConvNext model.
    Extracts features from specified layers and computes loss based on feature differences.
    
    Args:
        device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu')
        model_type (ConvNextType): Type of ConvNext model to use (tiny, small, base, large)
        feature_layers (List[int]): Indices of layers to extract features from
        feature_weights (Optional[List[float]]): Weights for each feature layer. If None, weights are computed with decay.
        use_gram (bool): Whether to use Gram matrix for style loss
        input_range (Tuple[float, float]): Expected input range for normalization
        layer_weight_decay (float): Decay factor for layer weights if feature_weights is None

        @misc{convnext_perceptual_loss2024,
        title={ConvNext Perceptual Loss: A Modern Perceptual Loss Implementation},
        author={Yipeng Sun},
        year={2024},
        publisher={GitHub},
        journal={GitHub repository},
        howpublished={url{https://github.com/sypsyp97/convnext_perceptual_loss}},
        doi={10.5281/zenodo.13991193}
}
        """
    def __init__(
        self, 
        device: torch.device,
        model_type: _ConvNextType = _ConvNextType.TINY,
        feature_layers: List[int] = [0, 2, 4, 6, 8, 10, 12, 14],
        feature_weights: Optional[List[float]] = None,
        use_gram: bool = True,
        input_range: Tuple[float, float] = (-1, 1),
        layer_weight_decay: float = 1.0
    ):
        """Initialize perceptual loss module"""
        super().__init__()
        
        self.device = device
        self.input_range = input_range
        self.use_gram = use_gram
        self.feature_layers = feature_layers
        
        # Calculate weights with decay if not specified
        if feature_weights is None:
            decay_values = [layer_weight_decay ** i for i in range(len(feature_layers))]
            weights = torch.tensor(decay_values, device=device, dtype=torch.float32)
            weights = weights / weights.sum()
        else:
            weights = torch.tensor(feature_weights, device=device, dtype=torch.float32)
        
        assert len(feature_layers) == len(weights), "Number of feature layers must match number of weights"
        self.register_buffer("feature_weights", weights)
        
        # Load pretrained ConvNext model
        model_name = f"convnext_{model_type.value}"
        try:
            weights_enum = getattr(models, f"ConvNeXt_{model_type.value.capitalize()}_Weights")
            weights = weights_enum.DEFAULT
            model = getattr(models, model_name)(weights=weights)
        except (AttributeError, ImportError):
            model = getattr(models, model_name)(pretrained=True)

        # Extract blocks and ensure they're in eval mode
        self.blocks = nn.ModuleList()
        for stage in model.features:
            if isinstance(stage, nn.Sequential):
                self.blocks.extend(stage)
            else:
                self.blocks.append(stage)
        
        self.blocks = self.blocks.eval().to(device)
        # Don't freeze parameters but set requires_grad=False since we don't update them
        for param in self.blocks.parameters():
            param.requires_grad_(False)
        
        # Register normalization parameters
        self.register_buffer(
            "mean", 
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", 
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        )
        
        self.to(device)

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor"""
        x = x.to(self.device)
        
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Ensure we create new leaf tensors while maintaining gradient flow
        x = x - torch.tensor(0., device=self.device)  # Create new leaf tensor
        
        min_val, max_val = self.input_range
        x = (x - min_val) / (max_val - min_val)
        mean = self.mean if isinstance(self.mean, torch.Tensor) else torch.tensor(self.mean, device=self.device)
        std = self.std if isinstance(self.std, torch.Tensor) else torch.tensor(self.std, device=self.device)
        x = (x - mean) / std
        
        if x.requires_grad:
            x.retain_grad()  # Retain gradients for intermediate values
            
        return x

    def gram_matrix(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Compute Gram matrix of feature maps"""
        b, c, h, w = x.size()
        features = x.view(b, c, -1)
        gram = torch.bmm(features, features.transpose(1, 2))
        if normalize:
            gram = gram / (c * h * w)
        return gram
    
    def compute_feature_loss(
        self, 
        input_features: List[torch.Tensor],
        target_features: List[torch.Tensor],
        layers_indices: List[int],
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute feature loss ensuring scalar output"""
        losses = []
        
        for idx, weight in zip(layers_indices, weights):
            input_feat = input_features[idx]
            target_feat = target_features[idx].detach()  # Detach target features
            
            if self.use_gram:
                input_gram = self.gram_matrix(input_feat)
                target_gram = self.gram_matrix(target_feat)
                layer_loss = nn.functional.l1_loss(input_gram, target_gram)
            else:
                layer_loss = nn.functional.mse_loss(input_feat, target_feat)
            
            losses.append(weight * layer_loss)
            
        # Sum all losses and ensure scalar output
        return torch.stack(losses).sum()

    def forward(
        self, 
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass to compute loss"""
        input = input.to(self.device)
        target = target.to(self.device)
        
        input = self.normalize_input(input)
        target = self.normalize_input(target)
        
        # Extract features
        input_features = []
        target_features = []
        
        x_input = input
        x_target = target
        for block in self.blocks:
            x_input = block(x_input)
            with torch.no_grad():  # No need to compute gradients for target features
                x_target = block(x_target)
            input_features.append(x_input)
            target_features.append(x_target)
        
        loss = self.compute_feature_loss(
            input_features, target_features,
            self.feature_layers, self.feature_weights # type: ignore
        )
        
        return loss


class FocalLoss(nn.Module):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    def __init__(self, alpha=0.1, gamma=2, reduction='mean', ignore_index=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # Apply ignore mask by removing elements entirely
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            if not mask.any():
                # All elements are ignored
                return torch.tensor(0.0, device=inputs.device, requires_grad=True)
            
            inputs = inputs[mask]
            targets = targets[mask]
        
        # Standard focal loss computation on filtered inputs
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Apply reduction
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction != "none":
            raise ValueError(
                f"Invalid value for arg 'reduction': '{self.reduction}' \n"
                f"Supported reduction modes: 'none', 'mean', 'sum'"
            )
        
        return loss

class WeightedMultiClassFocalLoss(nn.Module):
    """
    Multi-class Focal Loss with per-class alpha weighting.
    Useful when you have class imbalance.
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=None):
        super(WeightedMultiClassFocalLoss, self).__init__()
        self.alpha = alpha  # Should be a list/tensor of per-class weights
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C, H, W] - Raw logits from model
            targets: [N, H, W] - Ground truth class indices
        """
        # Standard cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, 
                                ignore_index=self.ignore_index,  # type: ignore
                                reduction='none')
        
        # Get class probabilities
        pt = F.softmax(inputs, dim=1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Handle ignored indices
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            pt = pt * mask.float()
            ce_loss = ce_loss * mask.float()
        
        # Apply focal weight
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply per-class alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha_tensor = torch.tensor(self.alpha, device=inputs.device)
            else:
                alpha_tensor = self.alpha
            
            # Get alpha values for each pixel based on its class
            alpha_t = alpha_tensor[targets]
            
            # Mask out ignored indices
            if self.ignore_index is not None:
                alpha_t = alpha_t * mask.float()
            
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.ignore_index is not None and self.reduction == 'mean':
            valid_elements = mask.sum()
            if valid_elements > 0:
                focal_loss = focal_loss.sum() / valid_elements
            else:
                focal_loss = focal_loss.sum()
        elif self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        
        return focal_loss
