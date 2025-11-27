import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
from .utils import *

class JaccardLoss(nn.Module):
    """
    Jaccard Loss (IoU Loss) for segmentation
    This loss directly optimizes the Intersection over Union metric
    Combined with Cross-Entropy/BCE loss for better gradient flow
    """
    def __init__(self, num_classes=1, ignore_index=255, smooth=1e-6, 
                 weight: Optional[torch.Tensor] = None, alpha=0.5, boundary_weight=0.0):
        """
        Args:
            num_classes: Number of segmentation classes
            ignore_index: Index to ignore in loss calculation (default: 255)
            smooth: Smoothing factor to avoid division by zero
            weight: Optional class weights for handling imbalance
            alpha: Weight between CE and Jaccard loss (0.5 = equal weight. Higher alpha = more CE focus)
            boundary_weight: Weight for Active Boundary Loss component (0.0 = no boundary loss)
        """
        super(JaccardLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.alpha = alpha
        
        if num_classes == 1:
            # Binary segmentation
            if weight is not None:
                if weight.shape[0] == 2:
                    weight = weight[1].float()
                elif weight.shape[0] > 2:
                    print(f"Warning: weight shape of {weight.shape} is invalid for binary classification, ignoring weight")
                    weight = None
            self.ce = nn.BCEWithLogitsLoss(pos_weight=weight, reduction='none')
            # store processed weight for later IoU/Jaccard computations
            self.iou_weight = torch.sqrt(weight).to(weight.device) if isinstance(weight, torch.Tensor) else None
        else:
            # Multi-class segmentation
            if weight is not None:
                if weight.shape[0] != num_classes:
                    print(f"Warning: weight shape of {weight.shape} is invalid for {num_classes} classes, ignoring weight")
                    weight = None
            self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')
            self.iou_weight = torch.sqrt(weight).to(weight.device) if isinstance(weight, torch.Tensor) else None

        if boundary_weight > 0.0:
            self.boundary_weight = boundary_weight
            self.boundary_loss = ActiveBoundaryLoss(num_classes=num_classes, ignore_index=ignore_index)
    
    def forward(self, logits, labels):
        """
        Args:
            logits: [B, C, H, W] - model logits
            labels: [B, H, W] or [B, 1, H, W] - ground truth class indices
        
        Returns:
            Combined loss: alpha * CE + (1 - alpha) * Jaccard
        """
        logits = logits.float()
        labels = labels.long()

        num_classes = logits.shape[1]
        # Ensure labels are [B, H, W]
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        
        # Create mask for valid pixels (not ignore_index)
        valid_mask = (labels != self.ignore_index)
        
        # --- Cross-Entropy / BCE Loss ---
        if num_classes > 1:
            ce_loss = self.ce(logits, labels)
            ce_loss = (ce_loss * valid_mask).sum() / (valid_mask.sum() + self.smooth)
        else:
            ce_loss = self.ce(logits.squeeze(1), labels.float())
            ce_loss = (ce_loss * valid_mask).sum() / (valid_mask.sum() + self.smooth)
        
        # --- Jaccard (IoU) Loss ---
        if num_classes == 1:
            # Binary segmentation
            probs = torch.sigmoid(logits.squeeze(1))
            
            # Apply mask
            probs_masked = probs * valid_mask
            targets_masked = labels.float() * valid_mask
            
            # Flatten
            probs_flat = probs_masked.view(-1)
            targets_flat = targets_masked.view(-1)
            
            # IoU calculation
            intersection = (probs_flat * targets_flat).sum()
            union = probs_flat.sum() + targets_flat.sum() - intersection
            iou = (intersection + self.smooth) / (union + self.smooth)
            
        else:
            # Multi-class segmentation
            probs = F.softmax(logits, dim=1)  # [B, C, H, W]
            
            # One-hot encode labels
            labels_one_hot = F.one_hot(labels.clamp(0, self.num_classes - 1), 
                                       num_classes=self.num_classes)  # [B, H, W, C]
            labels_one_hot = labels_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
            
            # Apply mask to both predictions and targets
            valid_mask_expanded = valid_mask.unsqueeze(1)  # [B, 1, H, W]
            probs_masked = probs * valid_mask_expanded
            labels_masked = labels_one_hot * valid_mask_expanded
            
            # Compute per-class IoU
            intersection = (probs_masked * labels_masked).sum(dim=(0, 2, 3))
            union = (probs_masked + labels_masked).sum(dim=(0, 2, 3)) - intersection
            iou_per_class = (intersection + self.smooth) / (union + self.smooth)
            
            # Average IoU across classes
            iou = (iou_per_class * self.iou_weight).sum() / (self.iou_weight.sum() + self.smooth) if self.iou_weight is not None else iou_per_class.mean()

        jaccard_loss = 1 - iou
        
        if self.boundary_weight > 0.0:
            boundary_loss_value = self.boundary_loss(logits, labels)
            if boundary_loss_value is not None and boundary_loss_value > 0:
                jaccard_loss = (1-self.boundary_weight) * jaccard_loss + self.boundary_weight * boundary_loss_value
        
        # Combined loss
        return self.alpha * ce_loss + (1 - self.alpha) * jaccard_loss

class DiceLoss(nn.Module):
    def __init__(self, num_classes=1, weights=None, smooth=1e-5, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        if num_classes == 1:
            if weights is not None and weights.shape[0] == 2:
                weights = weights[1].float()
            self.ce = nn.BCEWithLogitsLoss(pos_weight=weights) if weights is not None else nn.BCEWithLogitsLoss()
        else:
            self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)

    def forward(self, logits, labels):
        logits = logits.float()
        labels = labels.long()

        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)

        num_classes = logits.shape[1]
        
        # CE Loss - let PyTorch handle ignore_index
        if num_classes > 1:
            ce_loss = self.ce(logits, labels)
        else:
            valid_mask = (labels != self.ignore_index)
            ce_loss = self.ce(logits.squeeze(1), labels.float())
            ce_loss = (ce_loss * valid_mask).sum() / (valid_mask.sum() + self.smooth)
        
        # Dice Loss - mask out ignore_index
        valid_mask = (labels != self.ignore_index)
        
        if num_classes > 1:
            dice_loss = 0
            for i in range(num_classes):
                probs = torch.sigmoid(logits[:, i, :, :])
                target = ((labels == i) & valid_mask).float()
                probs_flat = probs.reshape(-1)
                target_flat = target.reshape(-1)
                intersection = (probs_flat * target_flat).sum()
                dice = (2. * intersection + self.smooth) / (probs_flat.sum() + target_flat.sum() + self.smooth)
                dice_loss += (1 - dice)
        else:
            probs = torch.sigmoid(logits.squeeze(1))
            target = (labels * valid_mask).float()
            intersection = (probs * target).sum()
            dice = (2. * intersection + self.smooth) / (probs.sum() + target.sum() + self.smooth)
            dice_loss = 1 - dice

        return 0.5 * dice_loss + 0.5 * ce_loss
    
class DiceJaccardLoss(nn.Module):
    def __init__(self, num_classes=1, weights=None, smooth=1e-5, ignore_index=255):
        super(DiceJaccardLoss, self).__init__()
        self.jaccard_loss = JaccardLoss(num_classes=num_classes, ignore_index=ignore_index, weight=weights, smooth=smooth, alpha=0.6)

    def forward(self, logits, labels):
        logits = logits.float()
        labels = labels.long()

        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        valid_mask = (labels != self.ignore_index)
        num_classes = logits.shape[1]
        
        if num_classes > 1:
            dice_loss = 0
            for i in range(num_classes):
                probs = torch.sigmoid(logits[:, i, :, :])
                target = ((labels == i) & valid_mask).float()
                probs_flat = probs.reshape(-1)
                target_flat = target.reshape(-1)
                intersection = (probs_flat * target_flat).sum()
                dice = (2. * intersection + self.smooth) / (probs_flat.sum() + target_flat.sum() + self.smooth)
                dice_loss += (1 - dice)
        else:
            probs = torch.sigmoid(logits.squeeze(1))
            target = (labels * valid_mask).float()
            intersection = (probs * target).sum()
            dice = (2. * intersection + self.smooth) / (probs.sum() + target.sum() + self.smooth)
            dice_loss = 1 - dice
        
        jaccard = self.jaccard_loss(logits, labels)
        return 0.3 * dice_loss + 0.7 * jaccard

class ActiveBoundaryLoss(nn.Module):
    """
    Multi-class compatible Active Boundary Loss. It helps sharpen boundaries by computing the KL divergence between them.
    This is especially hepful for thin structures.

    This is a vectorized implementation of the Active Boundary Loss described in:
    "Active Boundary Loss for Semantic Segmentation" https://arxiv.org/abs/2102.02696

    - logits: [B, C, H, W]
    - target: [B, H, W] with values in {0..num_classes-1} or ignore_index
    """
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        max_N_ratio: float = 1/100,
        max_clip_dist: float = 20.0,
        detach_kl: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.max_N_ratio = max_N_ratio
        self.max_clip_dist = max_clip_dist
        self.detach_kl = detach_kl

        # CE over 8 directions
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        # 9 offsets (8 dirs + center)
        self.offsets: torch.Tensor
        self.register_buffer(
            "offsets",
            torch.tensor([
                [ 1, 0],
                [-1, 0],
                [ 0,-1],
                [ 0, 1],
                [-1, 1],
                [ 1, 1],
                [-1,-1],
                [ 1,-1],
                [ 0, 0],  # center (ignored)
            ], dtype=torch.long)
        )


    # ------------------------------------------------------------
    # GT → boundary mask
    # ------------------------------------------------------------
    def gt2boundary(self, target):
        B, H, W = target.shape

        # vertical changes
        vert = torch.zeros_like(target, dtype=torch.bool)
        vert[:, 1:] = target[:, 1:] != target[:, :-1]

        # horizontal changes
        horiz = torch.zeros_like(target, dtype=torch.bool)
        horiz[:, :, 1:] = target[:, :, 1:] != target[:, :, :-1]

        boundary = vert | horiz
        boundary |= (target == self.ignore_index)

        return boundary.unsqueeze(1)     # [B,1,H,W]


    # ------------------------------------------------------------
    # Predicted KL-based boundary
    # ------------------------------------------------------------
    def logits2boundary(self, prob):
        B, C, H, W = prob.shape

        # compute pairwise KL spikes
        kl_ud = kl_div(prob[:, :, 1:, :], prob[:, :, :-1, :]).sum(1, keepdim=True)
        kl_lr = kl_div(prob[:, :, :, 1:], prob[:, :, :, :-1]).sum(1, keepdim=True)

        kl_map = F.pad(kl_ud, (0,0,0,1)) + F.pad(kl_lr, (0,1,0,0))

        # adaptive threshold
        eps = 1e-5
        max_N = H * W * self.max_N_ratio

        with torch.no_grad():
            while True:
                mask = kl_map > eps
                if mask.sum() > max_N:
                    eps *= 1.2
                else:
                    break

        # dilate to stabilize edges
        mask = mask.float()
        mask = F.max_pool2d(mask, 3, 1, 1) > 0


        return mask.squeeze(1)      # [B,H,W] bool


    # ------------------------------------------------------------
    # Fast signed distance via repeated 3×3 erosions (DT approx)
    # ------------------------------------------------------------
    def get_dist_maps(self, boundary_mask):
        """
        boundary_mask: [B,1,H,W] or [B,H,W]
        Returns [B,H,W] signed distances (negative inside).
        """

        if boundary_mask.dim() == 4:
            boundary_mask = boundary_mask[:,0]

        B, H, W = boundary_mask.shape
        device = boundary_mask.device

        # distance accumulates iterations
        dist = torch.zeros((B, H, W), device=device)

        # working mask: 1=inside, 0=boundary/holes
        active = (~boundary_mask).float()

        # iterate approx EDT (fixed 20 steps max)
        for i in range(1, 21):
            new_active = F.max_pool2d(active.unsqueeze(1), 3, 1, 1).squeeze(1)
            delta = (active > 0) & (new_active < active)
            if not delta.any():
                break
            active[delta] = 0
            dist[delta] = float(i)

        return dist


    # ------------------------------------------------------------
    # FULL LOSS FORWARD
    # ------------------------------------------------------------
    def forward(self, logits, target):
        B, C, H, W = logits.shape
        device = logits.device

        # resize if needed
        if logits.shape[2:] != target.shape[1:]:
            logits = F.interpolate(logits, target.shape[1:], mode="bilinear", align_corners=True)

        prob = logits.softmax(dim=1)

        gt_bnd  = self.gt2boundary(target)
        pred_bnd = self.logits2boundary(prob)

        # nothing predicted
        if pred_bnd.sum() == 0:
            return None

        # signed distance
        dist = self.get_dist_maps(gt_bnd)   # [B,H,W]

        # gather boundary coords (fix batch=1 issue)
        coords = torch.nonzero(pred_bnd, as_tuple=False)

        if coords.numel() == 0:
            return None

        # coords shape can be [K,2] or [K,3]
        if coords.shape[1] == 2:
            b_idx = torch.zeros(coords.size(0), dtype=torch.long, device=device)
            y_idx = coords[:,0]
            x_idx = coords[:,1]
        elif coords.shape[1] == 3:
            b_idx, y_idx, x_idx = coords.T
        else:
            raise RuntimeError(f"Unexpected coords shape {coords.shape}")

        K = coords.size(0)

        # offsets to neighbors
        off = self.offsets.to(device)    # [9,2]

        oy = (y_idx[:,None] + off[:,0]).clamp(0, H-1)
        ox = (x_idx[:,None] + off[:,1]).clamp(0, W-1)

        # distances for GT direction
        dist_vals = dist[b_idx[:,None], oy, ox]    # [K,9]
        dir_gt = torch.argmin(dist_vals, dim=1)
        valid = (dir_gt != 8)

        if not valid.any():
            return None

        dir_gt = dir_gt[valid]
        oy8 = oy[valid, :8]
        ox8 = ox[valid, :8]
        b_valid = b_idx[valid]
        y_valid = y_idx[valid]
        x_valid = x_idx[valid]

        # KL(center || neighbors)
        prob_hw = prob.permute(0,2,3,1)  # [B,H,W,C]

        center = prob_hw[b_valid, y_valid, x_valid]       # [K,C]
        neigh  = prob_hw[b_valid[:,None], oy8, ox8]       # [K,8,C]

        if self.detach_kl:
            neigh = neigh.detach()

        kl_vals = kl_div(center.unsqueeze(1).expand_as(neigh)+1e-8, neigh).sum(dim=2)  # [K,8]

        # distance weights
        w = -dist[b_valid, y_valid, x_valid]
        w = torch.clamp(w / self.max_clip_dist, 0, 1)

        # CE over 8 directions
        loss = self.criterion(kl_vals, dir_gt)
        loss = (loss * w).mean()

        return loss

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