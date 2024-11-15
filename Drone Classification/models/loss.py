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

# This loss checks how close a positive pixel is to a true positive. 
# Total positive count is used to check accuracy to mangrove size (Overall size is what we care about to monitor health).
# It also incorporates Jaccard loss to increase IOU
class DistanceCountLoss(nn.Module):
    def __init__(self, smooth=1e-10, weight_jaccard=0.9, weight_distance_count=0.1):
        super(DistanceCountLoss, self).__init__()
        self.max_samples = 100
        self.smooth = smooth
        self.weight_jaccard = weight_jaccard
        self.weight_distance_count = weight_distance_count

    def jaccard_loss(self, y_pred, y_true):
        # Apply sigmoid to predictions to get probabilities
        y_pred = torch.sigmoid(y_pred)

        # Flatten the tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # Calculate intersection and union
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum() - intersection

        # Jaccard index and loss
        jaccard_index = (intersection + self.smooth) / (union + self.smooth)
        return 1 - jaccard_index

    def chamfer_distance(self, pred_points, label_points):
        # Check if either set is empty
        if pred_points.shape[0] == 0:
            return label_points.shape[0]  # Penalize based on the count of positives in label
        if label_points.shape[0] == 0:
            return pred_points.shape[0]  # Penalize based on the count of positives in prediction

        # Compute distances only one way (pred -> label) at a time
        dist_pred_to_label = torch.cdist(pred_points, label_points).min(dim=1)[0]
        dist_label_to_pred = torch.cdist(label_points, pred_points).min(dim=1)[0]
        
        # Chamfer distance is the mean of both sets' minimum distances
        return dist_pred_to_label.mean() + dist_label_to_pred.mean()
    
    def sample_points(self, points):
        # Randomly sample points if there are more than max_samples
        if points.shape[0] > self.max_samples:
            indices = torch.randperm(points.shape[0])[:self.max_samples]
            points = points[indices]
        return points

    def forward(self, y_pred, y_true):
        # Calculate Jaccard loss
        jaccard_loss_value = self.jaccard_loss(y_pred, y_true)
        
        # Threshold to obtain positive pixels in predictions and labels
        pred_positives = self.sample_points((y_pred > 0.5).nonzero(as_tuple=False).float())
        label_positives = self.sample_points((y_true > 0.5).nonzero(as_tuple=False).float())

        # Count of positive pixels in label
        label_count = label_positives.shape[0]

        # If there are no positives in the labels, avoid distance calculation
        if label_count == 0:
            distance_loss = pred_positives.shape[0]  # Penalize only false positives
        else:
            # Calculate distance loss using Chamfer Distance
            distance_loss = self.chamfer_distance(pred_positives, label_positives)

        # Combined loss
        total_loss = (
            self.weight_jaccard * jaccard_loss_value
            + self.weight_distance_count * distance_loss
        )

        return total_loss


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
    

