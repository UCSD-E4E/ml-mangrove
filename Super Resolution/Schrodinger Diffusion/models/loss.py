import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

class BCEDiceLoss(nn.Module):
    def __init__(self, weight: Optional[torch.tensor] = None, size_average=True):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=weight, reduction='mean' if size_average else 'sum')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + 1) / (inputs.sum() + targets.sum() + 1)
        dice_loss = 1 - dice
        return bce_loss + dice_loss
    
class BCETverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-5, pos_weight=None):
        super(BCETverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, inputs, targets):
        # BCE Loss
        bce_loss = self.bce(inputs, targets)
        
        # Apply sigmoid to the inputs
        inputs = torch.sigmoid(inputs)
        
        # Flatten the tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # True positives, false positives, and false negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        
        # Tversky index
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        tversky_loss = 1 - Tversky
        
        return bce_loss + tversky_loss

class JaccardLoss(nn.Module):
    def __init__(self, smooth=1e-10):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred : torch.tensor, y_true: torch.tensor):
        y_pred = torch.sigmoid(y_pred)
        
        # Flatten the tensors to simplify the calculation
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        # Calculate intersection and union
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum() - intersection
        
        # Calculate the Jaccard index
        jaccard_index = (intersection + self.smooth) / (union + self.smooth)
        
        # Return the Jaccard loss (1 - Jaccard index)
        return 1 - jaccard_index

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
    def __init__(self, alpha=0.1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss