"""
attention.py — Attention Rollout for Vision-Transformer segmentation models.

Attention Rollout (Abnar & Zuidema, 2020) propagates attention matrices
layer-by-layer to produce a single spatial relevance map.

Strategy (tried in order):
  1. HuggingFace output_attentions=True  — works for SegFormer, Swin, BEiT, DeiT
  2. nn.MultiheadAttention forward hooks  — works for vanilla PyTorch ViTs
  3. Generic "attention-like" module hooks — last-resort name-based matching

Reference
---------
Abnar & Zuidema, "Quantifying Attention Flow in Transformers", ACL 2020.
"""
from __future__ import annotations

import inspect
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# Strategy 1 — HuggingFace output_attentions=True
# ──────────────────────────────────────────────────────────────

def _try_hf_attentions(
    model: nn.Module, input_tensor: torch.Tensor
) -> list[torch.Tensor] | None:
    """
    Call model(input_tensor, output_attentions=True) and collect any
    attention weight tensors returned in the output object.

    Returns a list of (B, H, N, N) tensors, or None if unsupported.
    """
    sig = inspect.signature(model.forward)
    if "output_attentions" not in sig.parameters:
        return None

    try:
        with torch.no_grad():
            out = model(input_tensor, output_attentions=True)
    except Exception:
        return None

    weights: list[torch.Tensor] = []

    # HF models return attentions as a tuple of per-layer tensors under
    # several attribute names depending on architecture.
    for attr in ("attentions", "encoder_attentions", "decoder_attentions",
                 "cross_attentions"):
        val = getattr(out, attr, None)
        if val is None and isinstance(out, dict):
            val = out.get(attr)
        if val is not None:
            for w in val:
                if isinstance(w, torch.Tensor) and w.ndim >= 3:
                    weights.append(w.float())

    return weights if weights else None


# ──────────────────────────────────────────────────────────────
# Strategy 2 — nn.MultiheadAttention hooks
# ──────────────────────────────────────────────────────────────

class _MHAHook:
    def __init__(self) -> None:
        self.weights: list[torch.Tensor] = []
        self._handles: list = []

    def register(self, model: nn.Module) -> "_MHAHook":
        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                self._handles.append(module.register_forward_hook(self._hook))
        return self

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _hook(self, module, inputs, output):
        attn_out, attn_w = output if isinstance(output, tuple) else (output, None)
        if attn_w is None:
            with torch.no_grad():
                q, k, v = inputs[0], inputs[1], inputs[2]
                try:
                    _, attn_w = module(q, k, v, need_weights=True,
                                       average_attn_weights=False)
                except TypeError:
                    _, attn_w = module(q, k, v, need_weights=True)
        if attn_w is not None:
            self.weights.append(attn_w.detach().float())


def _try_mha_hooks(
    model: nn.Module, input_tensor: torch.Tensor
) -> list[torch.Tensor] | None:
    hook = _MHAHook()
    hook.register(model)
    try:
        with torch.no_grad():
            model(input_tensor)
    finally:
        hook.remove()
    return hook.weights if hook.weights else None


# ──────────────────────────────────────────────────────────────
# Strategy 3 — Generic attention-module output hooks
# ──────────────────────────────────────────────────────────────

_ATTN_NAME_PATTERNS = ("attention", "selfattn", "selfattention", "attn")


def _looks_like_attn_module(module: nn.Module) -> bool:
    name = type(module).__name__.lower()
    return any(p in name for p in _ATTN_NAME_PATTERNS)


class _GenericAttnHook:
    """
    Hooks any module whose class name contains 'attention'.
    Captures the first square (N×N) tensor in the output tuple/list,
    which is almost always the attention weight matrix.
    """
    def __init__(self) -> None:
        self.weights: list[torch.Tensor] = []
        self._handles: list = []

    def register(self, model: nn.Module) -> "_GenericAttnHook":
        for module in model.modules():
            if _looks_like_attn_module(module) and not isinstance(module, nn.MultiheadAttention):
                self._handles.append(module.register_forward_hook(self._hook))
        return self

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _hook(self, module, inputs, output):
        candidates = output if isinstance(output, (tuple, list)) else [output]
        for c in candidates:
            if (isinstance(c, torch.Tensor) and c.ndim >= 3
                    and c.shape[-1] == c.shape[-2]):          # square → attention map
                self.weights.append(c.detach().float())
                break


def _try_generic_hooks(
    model: nn.Module, input_tensor: torch.Tensor
) -> list[torch.Tensor] | None:
    hook = _GenericAttnHook()
    hook.register(model)
    try:
        with torch.no_grad():
            model(input_tensor)
    finally:
        hook.remove()
    return hook.weights if hook.weights else None


# ──────────────────────────────────────────────────────────────
# Normalise attention tensors to (B, H, N, N)
# ──────────────────────────────────────────────────────────────

def _normalise_attn(w: torch.Tensor) -> torch.Tensor:
    """
    Coerce a captured tensor into (B, H, N, N) shape.
      - (B, H, N, N) → pass through
      - (B, N, N)    → unsqueeze head dim
      - (N, N)       → unsqueeze batch + head
    """
    if w.ndim == 4:
        return w
    if w.ndim == 3:
        return w.unsqueeze(1)   # add head dim
    if w.ndim == 2:
        return w.unsqueeze(0).unsqueeze(0)
    return w


# ──────────────────────────────────────────────────────────────
# Core rollout
# ──────────────────────────────────────────────────────────────

def _rollout(attention_matrices: list[torch.Tensor]) -> torch.Tensor:
    """
    Compute Attention Rollout from a list of (B, H, N, N) tensors.
    Returns (B, N, N).
    """
    result = None
    for attn in attention_matrices:
        attn = _normalise_attn(attn)
        attn_avg = attn.mean(dim=1)                          # (B, N, N)
        eye = torch.eye(attn_avg.shape[-1], device=attn_avg.device).unsqueeze(0)
        attn_aug = (attn_avg + eye) / 2.0
        attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        result = attn_aug if result is None else torch.bmm(attn_aug, result)
    return result   # (B, N, N)


# ──────────────────────────────────────────────────────────────
# Spatial map extraction
# ──────────────────────────────────────────────────────────────

def _to_spatial(attn_map_1d: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Reshape a 1-D token attribution vector to a 2-D spatial map.
    Falls back to a best-effort reshape for non-square token counts.
    """
    n = attn_map_1d.shape[0]
    H_in, W_in = input_tensor.shape[-2:]

    # Try exact square
    grid = int(n ** 0.5)
    if grid * grid == n:
        return attn_map_1d.reshape(grid, grid)

    # Try to find integer factors close to input aspect ratio
    aspect = W_in / max(H_in, 1)
    best_h, best_w, best_err = 1, n, float("inf")
    for h in range(1, n + 1):
        if n % h == 0:
            w = n // h
            err = abs(w / h - aspect)
            if err < best_err:
                best_err, best_h, best_w = err, h, w

    try:
        return attn_map_1d[: best_h * best_w].reshape(best_h, best_w)
    except RuntimeError:
        warnings.warn(f"Cannot reshape {n} tokens to 2-D; using 1-D map.")
        return attn_map_1d.unsqueeze(0)


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def attention_rollout(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
    cls_token: bool = False,
    output_size: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """
    Compute an Attention Rollout heatmap.

    Tries three strategies in order:
      1. HuggingFace ``output_attentions=True`` (SegFormer, Swin, BEiT …)
      2. ``nn.MultiheadAttention`` forward hooks (vanilla PyTorch ViTs)
      3. Generic name-based hooks on any module called '*attention*'

    Parameters
    ----------
    model : nn.Module
        Transformer-based segmentation model in eval mode.
    input_tensor : torch.Tensor
        Preprocessed image of shape (1, C, H, W).
    class_idx : int
        Target class (reserved for future class-conditional extensions).
    cls_token : bool
        Use CLS-token row (True) or mean over all tokens (False).
        SegFormer and Swin have no CLS token — keep False (default).
    output_size : (H, W), optional
        Resize heatmap to this resolution. Defaults to input spatial dims.

    Returns
    -------
    np.ndarray — float32 array in [0, 1], shape (H, W).
    """
    model.eval()

    # ── Collect attention matrices ──────────────────────────────
    weights = _try_hf_attentions(model, input_tensor)
    source = "HuggingFace output_attentions"

    if not weights:
        weights = _try_mha_hooks(model, input_tensor)
        source = "nn.MultiheadAttention hooks"

    if not weights:
        weights = _try_generic_hooks(model, input_tensor)
        source = "generic attention hooks"

    if not weights:
        raise RuntimeError(
            "No attention weights could be captured from this model.\n"
            "Tried: HuggingFace output_attentions=True, nn.MultiheadAttention hooks, "
            "and generic attention-module hooks.\n"
            "Options:\n"
            "  • Use method='gradcam' or 'lime' instead (both are model-agnostic).\n"
            "  • Subclass your model and expose attention weights explicitly."
        )

    # ── Rollout ─────────────────────────────────────────────────
    rollout = _rollout(weights)   # (1, N, N)

    if cls_token and rollout.shape[-1] > 1:
        attn_map = rollout[0, 0, 1:]    # CLS → patch
    else:
        attn_map = rollout[0].mean(dim=0)   # mean over all query tokens

    # ── Reshape to 2-D ──────────────────────────────────────────
    attn_map_2d = _to_spatial(attn_map, input_tensor)
    heatmap = attn_map_2d.cpu().float().numpy()

    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax > hmin:
        heatmap = (heatmap - hmin) / (hmax - hmin)

    if output_size is None:
        output_size = (input_tensor.shape[-2], input_tensor.shape[-1])
    h_t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    h_t = F.interpolate(h_t, size=output_size, mode="bilinear", align_corners=False)
    return h_t.squeeze().numpy().astype(np.float32)
