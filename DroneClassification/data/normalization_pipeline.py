"""
Simple Normalizer utilities for the Human Infrastructure project.

Gage review: Keep it minimal – one class with explicit methods
for each supported normalization.
"""

import numpy as np
from typing import Optional, Tuple


class Normalizer:
    """Collection of normalization methods.

    All methods operate on channel-first arrays (C, H, W) commonly used in
    our pipelines. Methods are intentionally simple and explicit.
    """

    # ImageNet statistics
    _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    _IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    @staticmethod
    def normalize_imagenet(data: np.ndarray) -> np.ndarray:
        """Normalize RGB image with ImageNet stats.

        Expects data shaped (3, H, W). If values are in 0-255, scales to 0-1
        before applying normalization.
        """
        if data.max() > 1.0:
            data = data / 255.0
        mean = Normalizer._IMAGENET_MEAN.reshape(3, 1, 1)
        std = Normalizer._IMAGENET_STD.reshape(3, 1, 1)
        return (data - mean) / std

    @staticmethod
    def denormalize_imagenet(normalized: np.ndarray) -> np.ndarray:
        """Invert ImageNet normalization and return uint8 image in 0-255."""
        mean = Normalizer._IMAGENET_MEAN.reshape(3, 1, 1)
        std = Normalizer._IMAGENET_STD.reshape(3, 1, 1)
        img01 = normalized * std + mean
        return np.clip(img01 * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def normalize_minmax(data: np.ndarray,
                         min_val: Optional[float] = None,
                         max_val: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
        """Min-max normalize to [0, 1]. Returns (normalized, min_val, max_val)."""
        if min_val is None:
            min_val = float(data.min())
        if max_val is None:
            max_val = float(data.max())
        denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
        return (data - min_val) / denom, min_val, max_val

    @staticmethod
    def denormalize_minmax(normalized: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """Invert min-max normalization given min and max used for scaling."""
        return normalized * (max_val - min_val) + min_val
