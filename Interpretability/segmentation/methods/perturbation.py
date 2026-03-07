"""
perturbation.py — Model-agnostic LIME and SHAP explanations for segmentation.

Both methods work by perturbing the input image (masking superpixel segments)
and observing how the model's class score changes.  They are slower than
gradient-based methods but fully model-agnostic — they work identically for
CNN and Transformer architectures.

Dependencies (optional — graceful ImportError messages if absent):
    pip install scikit-image shap
"""
from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────

def _get_superpixels(image_np: np.ndarray, n_segments: int = 50) -> np.ndarray:
    """Segment the image into superpixels using SLIC."""
    try:
        from skimage.segmentation import slic
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for LIME/SHAP: pip install scikit-image"
        ) from exc
    # image_np expected as (H, W, 3) float in [0, 1]
    segments = slic(image_np, n_segments=n_segments, compactness=10, start_label=0)
    return segments  # (H, W) int array


def _tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (1, C, H, W) tensor to a (H, W, C) float32 numpy array in [0,1]."""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    img = np.clip(img, 0, 1)
    return img


def _np_to_tensor(img_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert (H, W, C) float32 numpy to (1, C, H, W) tensor."""
    t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return t


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


def _class_score_fn(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
) -> float:
    """Return a scalar score for *class_idx* from the model output."""
    with torch.no_grad():
        output = _unwrap(model(input_tensor))
    if output.ndim == 4:    # (N, C, H, W)
        return output[0, class_idx].mean().item()
    elif output.ndim == 3:  # (N, H, W)
        return (output[0] == class_idx).float().mean().item()
    return output[class_idx].mean().item()


def _smooth(heatmap: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to soften superpixel block boundaries."""
    if sigma <= 0:
        return heatmap
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(heatmap, sigma=sigma).astype(np.float32)


def _mask_image(
    img_np: np.ndarray,
    segments: np.ndarray,
    active_mask: np.ndarray,
    background_value: float = 0.5,
) -> np.ndarray:
    """
    Return a copy of *img_np* where inactive superpixels are replaced by
    *background_value* (default: neutral grey).
    """
    masked = img_np.copy()
    for seg_id in np.unique(segments):
        if not active_mask[seg_id]:
            masked[segments == seg_id] = background_value
    return masked


# ──────────────────────────────────────────────────────────────
# LIME
# ──────────────────────────────────────────────────────────────

def lime_segmentation(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
    n_segments: int = 50,
    n_samples: int = 64,
    output_size: Optional[tuple[int, int]] = None,
    background_value: float = 0.5,
    random_seed: int = 42,
    smooth_sigma: float = 8.0,
) -> np.ndarray:
    """
    LIME explanation for a segmentation model (superpixel-based).

    Perturbs superpixels, fits a linear surrogate model, and returns the
    signed coefficient map as a heatmap.

    Parameters
    ----------
    model : nn.Module
        Segmentation model in eval mode.
    input_tensor : torch.Tensor
        Preprocessed image (1, C, H, W).
    class_idx : int
        Segmentation class to explain.
    n_segments : int
        Number of SLIC superpixels.
    n_samples : int
        Number of perturbation samples (higher = more accurate, slower).
    output_size : (H, W), optional
        Resize output heatmap. Defaults to input spatial dims.
    background_value : float
        Fill value for masked superpixels (default 0.5 = grey).
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray — float32 in [0, 1], shape (H, W).
    """
    rng = np.random.default_rng(random_seed)
    device = next(model.parameters()).device
    model.eval()

    img_np = _tensor_to_np(input_tensor)      # (H, W, C)
    segments = _get_superpixels(img_np, n_segments)
    num_segs = segments.max() + 1

    # ── Build perturbation dataset ──────────────────────────────
    X = np.zeros((n_samples, num_segs), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        active = rng.integers(0, 2, size=num_segs).astype(bool)
        masked = _mask_image(img_np, segments, active, background_value)
        tensor = _np_to_tensor(masked, device)
        score = _class_score_fn(model, tensor, class_idx)
        X[i] = active.astype(np.float32)
        y[i] = score

    # ── Fit linear surrogate ────────────────────────────────────
    try:
        from sklearn.linear_model import Ridge
        clf = Ridge(alpha=1.0)
        clf.fit(X, y)
        coefs = clf.coef_  # (num_segs,)
    except ImportError:
        warnings.warn("scikit-learn not found; falling back to lstsq for LIME.")
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    # ── Map coefficients back to pixel space ────────────────────
    heatmap = np.zeros(segments.shape, dtype=np.float32)
    for seg_id in range(num_segs):
        heatmap[segments == seg_id] = coefs[seg_id]

    heatmap = _smooth(heatmap, smooth_sigma)

    # Normalise to [0, 1]
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax > hmin:
        heatmap = (heatmap - hmin) / (hmax - hmin)

    if output_size is None:
        output_size = (input_tensor.shape[-2], input_tensor.shape[-1])
    h_t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    h_t = F.interpolate(h_t, size=output_size, mode="bilinear", align_corners=False)
    return h_t.squeeze().numpy().astype(np.float32)


# ──────────────────────────────────────────────────────────────
# SHAP (KernelSHAP via the shap library, with local fallback)
# ──────────────────────────────────────────────────────────────

def _build_predict_fn(
    model: torch.nn.Module,
    segments: np.ndarray,
    img_np: np.ndarray,
    class_idx: int,
    background_value: float,
    device: torch.device,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a numpy-in → numpy-out prediction function for KernelSHAP."""
    def predict(masks: np.ndarray) -> np.ndarray:
        # masks : (n_samples, num_segs) bool/int array
        scores = []
        for row in masks:
            masked = _mask_image(img_np, segments, row.astype(bool), background_value)
            tensor = _np_to_tensor(masked, device)
            scores.append(_class_score_fn(model, tensor, class_idx))
        return np.array(scores, dtype=np.float32).reshape(-1, 1)
    return predict


def shap_segmentation(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
    n_segments: int = 50,
    n_samples: int = 64,
    output_size: Optional[tuple[int, int]] = None,
    background_value: float = 0.5,
    random_seed: int = 42,
    smooth_sigma: float = 8.0,
) -> np.ndarray:
    """
    KernelSHAP explanation for a segmentation model.

    Uses the ``shap`` library if available, otherwise falls back to a
    permutation-sampling approximation (Štrumbelj & Kononenko, 2014).

    Parameters mirror those of :func:`lime_segmentation`.

    Returns
    -------
    np.ndarray — float32 in [0, 1], shape (H, W).
    """
    device = next(model.parameters()).device
    model.eval()

    img_np = _tensor_to_np(input_tensor)
    segments = _get_superpixels(img_np, n_segments)
    num_segs = segments.max() + 1

    predict_fn = _build_predict_fn(model, segments, img_np, class_idx, background_value, device)
    baseline = np.zeros((1, num_segs), dtype=np.float32)     # all masked
    foreground = np.ones((1, num_segs), dtype=np.float32)    # all visible

    try:
        import shap  # optional dependency
        explainer = shap.KernelExplainer(predict_fn, baseline)
        shap_values = explainer.shap_values(foreground, nsamples=n_samples, silent=True)
        coefs = np.array(shap_values).flatten()[:num_segs]

    except ImportError:
        warnings.warn(
            "shap library not installed; using permutation-sampling approximation. "
            "Install with: pip install shap"
        )
        coefs = _permutation_shap(predict_fn, num_segs, n_samples, random_seed)

    # ── Pixel-space heatmap ─────────────────────────────────────
    heatmap = np.zeros(segments.shape, dtype=np.float32)
    for seg_id in range(num_segs):
        heatmap[segments == seg_id] = coefs[seg_id] if seg_id < len(coefs) else 0.0

    heatmap = _smooth(heatmap, smooth_sigma)

    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax > hmin:
        heatmap = (heatmap - hmin) / (hmax - hmin)

    if output_size is None:
        output_size = (input_tensor.shape[-2], input_tensor.shape[-1])
    h_t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    h_t = F.interpolate(h_t, size=output_size, mode="bilinear", align_corners=False)
    return h_t.squeeze().numpy().astype(np.float32)


def _permutation_shap(
    predict_fn: Callable,
    num_features: int,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """
    Minimal permutation-sampling SHAP approximation (no external library).

    Returns
    -------
    np.ndarray of shape (num_features,) — Shapley value estimates.
    """
    rng = np.random.default_rng(seed)
    phi = np.zeros(num_features, dtype=np.float64)

    for _ in range(n_samples):
        perm = rng.permutation(num_features)
        coalition = np.zeros(num_features, dtype=np.float32)

        prev_val = predict_fn(coalition.reshape(1, -1))[0, 0]
        for feat in perm:
            coalition[feat] = 1.0
            new_val = predict_fn(coalition.reshape(1, -1))[0, 0]
            phi[feat] += new_val - prev_val
            prev_val = new_val

    return (phi / n_samples).astype(np.float32)
