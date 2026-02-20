"""
gradcam.py — GradCAM and GradCAM++ for segmentation models (CNN-based).

Both methods register forward/backward hooks on a target Conv2d layer,
aggregate the resulting feature maps, and produce a spatial relevance map
that can be overlaid on the original image.

References
----------
* Selvaraju et al., "Grad-CAM", ICCV 2017.
* Chattopadhyay et al., "Grad-CAM++", WACV 2018.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _HookManager:
    """Thin context-manager wrapper around PyTorch forward/backward hooks."""

    def __init__(self, layer: nn.Module) -> None:
        self.layer = layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_handle = None
        self._bwd_handle = None

    def __enter__(self):
        self._fwd_handle = self.layer.register_forward_hook(self._save_activations)
        self._bwd_handle = self.layer.register_full_backward_hook(self._save_gradients)
        return self

    def __exit__(self, *_):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def _save_activations(self, _, __, output):
        self.activations = output.detach()

    def _save_gradients(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()


def _unwrap(out):
    """Extract a raw tensor from plain tensors or HuggingFace ModelOutput objects."""
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("logits", "last_hidden_state", "pred_masks", "masks_queries_logits"):
        if hasattr(out, attr) and isinstance(getattr(out, attr), torch.Tensor):
            return getattr(out, attr)
    if isinstance(out, dict):
        for key in ("logits", "last_hidden_state", "out", "output"):
            if key in out and isinstance(out[key], torch.Tensor):
                return out[key]
    if isinstance(out, (list, tuple)):
        for item in out:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Cannot extract tensor from {type(out).__name__}")


def _class_score(output: torch.Tensor, class_idx: int) -> torch.Tensor:
    """
    Derive a scalar class score from a segmentation output (N, C, H, W).
    We use the mean of the class channel over the spatial dimensions so the
    backward pass distributes gradients across the whole feature map.
    """
    output = _unwrap(output)
    if output.ndim == 4:          # (N, C, H, W) — dense logits
        return output[:, class_idx].mean()
    elif output.ndim == 3:        # (N, H, W) — single-channel / argmax
        return output[:, :, class_idx].mean()
    else:
        return output[class_idx].mean()


def gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    class_idx: int,
    output_size: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """
    Compute a GradCAM heatmap.

    Parameters
    ----------
    model : nn.Module
        A segmentation model in eval mode.
    input_tensor : torch.Tensor
        Preprocessed image tensor of shape (1, C, H, W).
    target_layer : nn.Module
        The Conv2d layer to hook into.
    class_idx : int
        The segmentation class to explain.
    output_size : (H, W), optional
        Resize the heatmap to this size. Defaults to input spatial dims.

    Returns
    -------
    np.ndarray — float32 array in [0, 1], shape (H, W).
    """
    model.eval()
    input_tensor = input_tensor.requires_grad_(True)

    with _HookManager(target_layer) as hooks:
        output = model(input_tensor)
        score = _class_score(output, class_idx)
        model.zero_grad()
        score.backward()

        acts = hooks.activations   # (1, K, h, w)
        grads = hooks.gradients    # (1, K, h, w)

    # Global average pool the gradients over spatial dims → weights (1, K, 1, 1)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)       # (1, 1, h, w)
    cam = F.relu(cam)

    # Resize to input/requested resolution
    if output_size is None:
        output_size = (input_tensor.shape[-2], input_tensor.shape[-1])
    cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)

    heatmap = cam.squeeze().cpu().numpy().astype(np.float32)
    # Normalise to [0, 1]
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax > hmin:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    return heatmap


def gradcam_plus_plus(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    class_idx: int,
    output_size: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """
    Compute a GradCAM++ heatmap (higher-order gradient weighting).

    Parameters mirror those of :func:`gradcam`.

    Returns
    -------
    np.ndarray — float32 array in [0, 1], shape (H, W).
    """
    model.eval()
    input_tensor = input_tensor.requires_grad_(True)

    with _HookManager(target_layer) as hooks:
        output = model(input_tensor)
        score = _class_score(output, class_idx)
        model.zero_grad()
        score.backward()

        acts = hooks.activations   # (1, K, h, w)
        grads = hooks.gradients    # (1, K, h, w)

    # GradCAM++ weight computation
    grads_sq = grads ** 2
    grads_cb = grads ** 3
    denom = 2 * grads_sq + acts * grads_cb.sum(dim=(2, 3), keepdim=True)
    denom = torch.where(denom != 0, denom, torch.ones_like(denom))
    alpha = grads_sq / denom                             # (1, K, h, w)
    weights = (alpha * F.relu(grads)).mean(dim=(2, 3), keepdim=True)  # (1, K, 1, 1)

    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    if output_size is None:
        output_size = (input_tensor.shape[-2], input_tensor.shape[-1])
    cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)

    heatmap = cam.squeeze().cpu().numpy().astype(np.float32)
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax > hmin:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    return heatmap
