import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
from .utils import LabelSmoothSoftmaxCEV1 as LSSCE
from .utils import *
from torchvision import transforms
from functools import partial
from operator import itemgetter
    
class JaccardLoss(nn.Module):
    """
    Jaccard Loss (IoU Loss) for segmentation
    This loss directly optimizes the Intersection over Union metric
    Combined with Cross-Entropy/BCE loss for better gradient flow
    """
    def __init__(self, num_classes=1, ignore_index=255, smooth=1e-6, 
                 weight: Optional[torch.Tensor] = None, alpha=0.5):
        """
        Args:
            num_classes: Number of segmentation classes
            ignore_index: Index to ignore in loss calculation (default: 255)
            smooth: Smoothing factor to avoid division by zero
            weight: Optional class weights for handling imbalance
            alpha: Weight between CE and Jaccard loss (0.5 = equal weight)
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
        else:
            # Multi-class segmentation
            if weight is not None:
                if weight.shape[0] != num_classes:
                    print(f"Warning: weight shape of {weight.shape} is invalid for {num_classes} classes, ignoring weight")
                    weight = None
            self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')
    
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
        # Ensure labels are [B, H, W]
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        
        # Create mask for valid pixels (not ignore_index)
        valid_mask = (labels != self.ignore_index)
        
        # --- Cross-Entropy / BCE Loss ---
        if self.num_classes > 1:
            ce_loss = self.ce(logits, labels)
            ce_loss = (ce_loss * valid_mask).sum() / (valid_mask.sum() + self.smooth)
        else:
            ce_loss = self.ce(logits.squeeze(1), labels.float())
            ce_loss = (ce_loss * valid_mask).sum() / (valid_mask.sum() + self.smooth)
        
        # --- Jaccard (IoU) Loss ---
        if self.num_classes == 1:
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
            iou = iou_per_class.mean()
        
        jaccard_loss = 1 - iou
        
        # Combined loss
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * jaccard_loss
        
        return total_loss

class ActiveBoundaryLoss(nn.Module):
    """
    Loss function that focuses on the boundaries of segmented regions: https://github.com/wangchi95/active-boundary-loss
    It computes the KL divergence between predicted and ground truth boundaries, weighted by the distance to the nearest boundary.
    
    This encourages the model to pay more attention to the edges of boundary regions during training.
    """
    def __init__(self, isdetach=True, max_N_ratio = 1/100, ignore_label = 255, label_smoothing=0.2, weight = None, max_clip_dist = 20.):
        super(ActiveBoundaryLoss, self).__init__()
        self.ignore_label = ignore_label
        self.label_smoothing = label_smoothing
        self.isdetach=isdetach
        self.max_N_ratio = max_N_ratio

        self.weight_func = lambda w, max_distance=max_clip_dist: torch.clamp(w, max=max_distance) / max_distance

        self.dist_map_transform = transforms.Compose([
            lambda img: img.unsqueeze(0),
            lambda nd: nd.type(torch.int64),
            partial(class2one_hot, C=1),
            itemgetter(0),
            lambda t: t.cpu().numpy(),
            one_hot2dist,
            lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])

        if label_smoothing == 0:
            self.criterion = nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=ignore_label,
                reduction='none'
            )
        else:
            self.criterion = LSSCE(
                reduction='none',
                ignore_index=ignore_label,
                lb_smooth = label_smoothing
            )

    def logits2boundary(self, logit):
        eps = 1e-5
        _, _, h, w = logit.shape
        max_N = (h*w) * self.max_N_ratio
        kl_ud = kl_div(logit[:, :, 1:, :], logit[:, :, :-1, :]).sum(1, keepdim=True)
        kl_lr = kl_div(logit[:, :, :, 1:], logit[:, :, :, :-1]).sum(1, keepdim=True)
        kl_ud = torch.nn.functional.pad(
            kl_ud, [0, 0, 0, 1, 0, 0, 0, 0], mode='constant', value=0)
        kl_lr = torch.nn.functional.pad(
            kl_lr, [0, 1, 0, 0, 0, 0, 0, 0], mode='constant', value=0)
        kl_combine = kl_lr+kl_ud
        while True: # avoid the case that full image is the same color
            kl_combine_bin = (kl_combine > eps).to(torch.float)
            if kl_combine_bin.sum() > max_N:
                eps *=1.2
            else:
                break
        #dilate
        dilate_weight = torch.ones((1,1,3,3)).cuda()
        edge2 = torch.nn.functional.conv2d(kl_combine_bin, dilate_weight, stride=1, padding=1)
        edge2 = edge2.squeeze(1)  # NCHW->NHW
        kl_combine_bin = (edge2 > 0)
        return kl_combine_bin

    def gt2boundary(self, gt, ignore_label=-1):  # gt NHW

        # Handle both [B, H, W] and [B, C, H, W]
        if gt.dim() == 3:
            # [B, H, W] case
            gt_ud = gt[:, 1:, :] - gt[:, :-1, :]
            gt_lr = gt[:, :, 1:] - gt[:, :, :-1]
            gt_ud = F.pad(gt_ud, [0, 0, 0, 1], value=0) != 0
            gt_lr = F.pad(gt_lr, [0, 1, 0, 0], value=0) != 0
        elif gt.dim() == 4:
            # [B, C, H, W] case
            gt_ud = gt[:, :, 1:, :] - gt[:, :, :-1, :]
            gt_lr = gt[:, :, :, 1:] - gt[:, :, :, :-1]
            gt_ud = F.pad(gt_ud, [0, 0, 0, 1], value=0) != 0
            gt_lr = F.pad(gt_lr, [0, 1, 0, 0], value=0) != 0
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {gt.dim()}D with shape {gt.shape}")
   
        gt_combine = gt_lr+gt_ud
        del gt_lr
        del gt_ud
        
        # set 'ignore area' to all boundary
        gt_combine += (gt==ignore_label)
        
        return gt_combine > 0

    def get_direction_gt_predkl(self, pred_dist_map, pred_bound, logits):
        # NHW,NHW,NCHW
        eps = 1e-5
        # bound = torch.where(pred_bound)  # 3k
        bound = torch.nonzero(pred_bound*1)
        n,x,y = bound.T
        max_dis = 1e5

        logits = logits.permute(0,2,3,1) # NHWC

        pred_dist_map_d = torch.nn.functional.pad(pred_dist_map,(1,1,1,1,0,0),mode='constant', value=max_dis) # NH+2W+2

        logits_d = torch.nn.functional.pad(logits,(0,0,1,1,1,1,0,0),mode='constant') # N(H+2)(W+2)C
        logits_d[:,0,:,:] = logits_d[:,1,:,:] # N(H+2)(W+2)C
        logits_d[:,-1,:,:] = logits_d[:,-2,:,:] # N(H+2)(W+2)C
        logits_d[:,:,0,:] = logits_d[:,:,1,:] # N(H+2)(W+2)C
        logits_d[:,:,-1,:] = logits_d[:,:,-2,:] # N(H+2)(W+2)C
        
        """
        | 4| 0| 5|
        | 2| 8| 3|
        | 6| 1| 7|
        """
        x_range = [1, -1,  0, 0, -1,  1, -1,  1, 0]
        y_range = [0,  0, -1, 1,  1,  1, -1, -1, 0]
        dist_maps = torch.zeros((0,len(x))).cuda() # 8k
        kl_maps = torch.zeros((0,len(x))).cuda() # 8k

        kl_center = logits[(n,x,y)] # KC

        for dx, dy in zip(x_range, y_range):
            dist_now = pred_dist_map_d[(n,x+dx+1,y+dy+1)]
            dist_maps = torch.cat((dist_maps,dist_now.unsqueeze(0)),0)

            if dx != 0 or dy != 0:
                logits_now = logits_d[(n,x+dx+1,y+dy+1)]
                # kl_map_now = torch.kl_div((kl_center+eps).log(), logits_now+eps).sum(2)  # 8KC->8K
                if self.isdetach:
                    logits_now = logits_now.detach()
                kl_map_now = kl_div(kl_center, logits_now)
                
                kl_map_now = kl_map_now.sum(1)  # KC->K
                kl_maps = torch.cat((kl_maps,kl_map_now.unsqueeze(0)),0)
                torch.clamp(kl_maps, min=0.0, max=20.0)

        # direction_gt shound be Nk  (8k->K)
        direction_gt = torch.argmin(dist_maps, dim=0)
        # weight_ce = pred_dist_map[bound]
        weight_ce = pred_dist_map[(n,x,y)]
        # print(weight_ce)

        # delete if min is 8 (local position)
        direction_gt_idx = [direction_gt!=8]
        direction_gt = direction_gt[direction_gt_idx]


        kl_maps = torch.transpose(kl_maps,0,1)
        direction_pred = kl_maps[direction_gt_idx]
        weight_ce = weight_ce[direction_gt_idx]

        return direction_gt, direction_pred, weight_ce

    def get_dist_maps(self, target):
        target_detach = target.clone().detach()
        dist_maps = torch.cat([self.dist_map_transform(target_detach[i]) for i in range(target_detach.shape[0])])
        out = -dist_maps
        out = torch.where(out>0, out, torch.zeros_like(out))
        
        return out

    def forward(self, logits, target):
        eps = 1e-10
        ph, pw = logits.size(2), logits.size(3)
        h, w = target.size(1), target.size(2)

        if ph != h or pw != w:
            logits = F.interpolate(input=logits, size=(
                h, w), mode='bilinear', align_corners=True)

        gt_boundary = self.gt2boundary(target, ignore_label=self.ignore_label)

        dist_maps = self.get_dist_maps(gt_boundary).cuda() # <-- it will slow down the training, you can put it to dataloader.

        pred_boundary = self.logits2boundary(logits)
        if pred_boundary.sum() < 1: # avoid nan
            return None # you should check in the outside. if None, skip this loss.
        
        direction_gt, direction_pred, weight_ce = self.get_direction_gt_predkl(dist_maps, pred_boundary, logits) # NHW,NHW,NCHW

        # direction_pred [K,8], direction_gt [K]
        loss = self.criterion(direction_pred, direction_gt) # careful
        
        weight_ce = self.weight_func(weight_ce)
        loss = (loss * weight_ce).mean()  # add distance weight

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