"""
integrated_gradients.py — Integrated Gradients via Captum.

Integrated Gradients (Sundararajan et al., 2017) computes per-pixel
attributions by integrating gradients along a straight path from a
baseline (black image) to the actual input.  Compared to LIME/SHAP it is:

  • Natively smooth — pixel-level, no superpixels needed
  • Fast — single forward+backward sweep (n_steps passes)
  • Works for both CNN and Transformer architectures

Reference
---------
Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _unwrap(out):
    """Extract a raw tensor from plain tensors or HuggingFace ModelOutput."""
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


def _forward_fn(model: nn.Module, class_idx: int):
    """Return a scalar-output function suitable for Captum."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        out = _unwrap(model(x))
        if out.ndim == 4:           # (N, C, H, W)
            return out[:, class_idx].mean(dim=(1, 2))   # (N,)
        elif out.ndim == 3:         # (N, H, W)
            return (out == class_idx).float().mean(dim=(1, 2))
        return out[:, class_idx]
    return fn


def integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
    n_steps: int = 25,
    baseline: Optional[torch.Tensor] = None,
    output_size: Optional[tuple[int, int]] = None,
    smooth_sigma: float = 2.0,
) -> np.ndarray:
    """
    Compute an Integrated Gradients attribution heatmap.

    Parameters
    ----------
    model : nn.Module
        Segmentation model in eval mode.
    input_tensor : torch.Tensor
        Preprocessed image of shape (1, C, H, W).
    class_idx : int
        Segmentation class to explain.
    n_steps : int
        Number of interpolation steps along the baseline→input path.
        More steps = more accurate, slightly slower.  50 is usually enough.
    baseline : torch.Tensor, optional
        Reference input of the same shape as *input_tensor*.
        Defaults to an all-zeros (black) image.
    output_size : (H, W), optional
        Resize heatmap to this resolution. Defaults to input spatial dims.

    Returns
    -------
    np.ndarray — float32 in [0, 1], shape (H, W).
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise ImportError(
            "captum is required for Integrated Gradients: pip install captum"
        ) from exc

    model.eval()

    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    ig = IntegratedGradients(_forward_fn(model, class_idx))

    # Captum needs requires_grad on input
    attributions = ig.attribute(
        input_tensor,
        baselines=baseline,
        n_steps=n_steps,
        return_convergence_delta=False,
    )   # (1, C, H, W)

    # Collapse channels: sum of absolute attributions → (H, W)
    # Using abs before summing ensures both positive and negative
    # attributions contribute to the prominence map.
    heatmap = attributions.squeeze(0).abs().sum(dim=0)
    heatmap = heatmap.detach().cpu().float().numpy()

    # Clip to 99th percentile before normalising so a handful of extreme
    # outlier pixels don't compress the rest of the map into near-zero.
    p99 = float(np.percentile(heatmap, 99))
    if p99 > 0:
        heatmap = np.clip(heatmap, 0, p99)

    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap, sigma=smooth_sigma).astype(np.float32)

    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax > hmin:
        heatmap = (heatmap - hmin) / (hmax - hmin)

    if output_size is None:
        output_size = (input_tensor.shape[-2], input_tensor.shape[-1])
    h_t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    h_t = F.interpolate(h_t, size=output_size, mode="bilinear", align_corners=False)
    return h_t.squeeze().numpy().astype(np.float32)
