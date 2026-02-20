"""
validation.py — Input/output validation for seg_interpret.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────
# Custom exception hierarchy
# ──────────────────────────────────────────────

class SegInterpretError(Exception):
    """Base error for seg_interpret."""


class InvalidImageError(SegInterpretError):
    """Raised when the supplied image path or array is invalid."""


class InvalidModelError(SegInterpretError):
    """Raised when the model fails compatibility checks."""


class InvalidMethodError(SegInterpretError):
    """Raised when an unsupported method is requested."""


class InvalidClassIndexError(SegInterpretError):
    """Raised when the requested class index is out of range."""


# ──────────────────────────────────────────────
# Image validation
# ──────────────────────────────────────────────

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
SUPPORTED_METHODS = {"auto", "gradcam", "gradcam++", "attention_rollout", "lime", "shap", "integrated_gradients"}


def validate_image_path(path: Union[str, Path]) -> Path:
    """Ensure the path points to a readable, supported image file."""
    path = Path(path)
    if not path.exists():
        raise InvalidImageError(f"Image file not found: {path}")
    if not path.is_file():
        raise InvalidImageError(f"Path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise InvalidImageError(
            f"Unsupported image extension '{path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )
    if os.path.getsize(path) == 0:
        raise InvalidImageError(f"Image file is empty: {path}")
    return path


def validate_image_tensor(tensor: torch.Tensor, name: str = "image") -> None:
    """Validate a preprocessed image tensor (N, C, H, W)."""
    if not isinstance(tensor, torch.Tensor):
        raise InvalidImageError(f"'{name}' must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim not in (3, 4):
        raise InvalidImageError(f"'{name}' must be 3-D (C,H,W) or 4-D (N,C,H,W), got shape {tuple(tensor.shape)}")
    if tensor.ndim == 4 and tensor.shape[0] != 1:
        raise InvalidImageError(f"Batch size must be 1 for interpretation; got {tensor.shape[0]}")
    if tensor.shape[-2] < 4 or tensor.shape[-1] < 4:
        raise InvalidImageError(f"Image spatial dimensions too small: {tuple(tensor.shape[-2:])}")


# ──────────────────────────────────────────────
# Model validation
# ──────────────────────────────────────────────

def validate_model(model: nn.Module) -> None:
    """Basic checks that the object is a usable PyTorch model."""
    if not isinstance(model, nn.Module):
        raise InvalidModelError(
            f"model must be a torch.nn.Module, got {type(model).__name__}"
        )
    param_count = sum(1 for _ in model.parameters())
    if param_count == 0:
        raise InvalidModelError("Model has no parameters — it may not be initialised correctly.")


def validate_model_output(
    output: torch.Tensor,
    expected_classes: Optional[int],
    image_shape: tuple,
) -> None:
    """
    Validate segmentation model output.

    Expected shape: (N, C, H, W)  [dense logits]
    or             (N, H, W)       [single-class mask]
    """
    if not isinstance(output, torch.Tensor):
        raise InvalidModelError(
            f"Model output must be a torch.Tensor, got {type(output).__name__}. "
            "Ensure the model returns raw logits / a single tensor."
        )
    if output.ndim not in (3, 4):
        raise InvalidModelError(
            f"Expected model output to be 3-D or 4-D, got shape {tuple(output.shape)}."
        )
    if output.ndim == 4 and expected_classes is not None:
        if output.shape[1] != expected_classes:
            raise InvalidModelError(
                f"Model output has {output.shape[1]} channels but num_classes={expected_classes}."
            )


# ──────────────────────────────────────────────
# Method / class-index validation
# ──────────────────────────────────────────────

def validate_method(method: str) -> str:
    method = method.lower().strip()
    if method not in SUPPORTED_METHODS:
        raise InvalidMethodError(
            f"Unknown method '{method}'. Supported: {sorted(SUPPORTED_METHODS)}"
        )
    return method


def validate_class_index(class_idx: Optional[int], num_classes: int) -> None:
    if class_idx is None:
        return  # will default to argmax
    if not isinstance(class_idx, int):
        raise InvalidClassIndexError(f"class_idx must be an int, got {type(class_idx).__name__}")
    if class_idx < 0 or class_idx >= num_classes:
        raise InvalidClassIndexError(
            f"class_idx={class_idx} is out of range for num_classes={num_classes}."
        )
