"""
detect.py — Heuristic auto-detection of CNN vs. Transformer segmentation models.
"""
from __future__ import annotations

import torch.nn as nn


# Module type names that strongly suggest a Vision Transformer backbone
_TRANSFORMER_INDICATORS = {
    # layer/module class names (lowercase)
    "attention", "multiheadattention", "selfattention", "transformer",
    "encoderlayer", "transformerlayer", "windowattention",  # Swin
    "segformer", "segformerencoder", "mixvision",          # SegFormer
    "vit", "vitblock", "vitlayer",                         # ViT
    "beit", "deit",
    "deformableattention",                                 # Mask2Former / DINO
    "crossattention",
}

# Module type names that strongly suggest a convolutional backbone
_CNN_INDICATORS = {
    "conv2d", "convbnrelu", "convbn", "depthwiseconv",
    "residualblock", "bottleneck", "basicblock",           # ResNet
    "denseblock", "denselayer",                            # DenseNet
    "asppconv", "aspp",                                    # DeepLab
    "separableconv2d",
}


def _module_names(model: nn.Module) -> set[str]:
    """Return a set of lowercased class names for all submodules."""
    return {type(m).__name__.lower() for m in model.modules()}


def detect_model_type(model: nn.Module) -> str:
    """
    Heuristically determine whether *model* is CNN- or Transformer-based.

    Returns
    -------
    'cnn' | 'transformer' | 'unknown'
    """
    names = _module_names(model)

    transformer_score = len(names & _TRANSFORMER_INDICATORS)
    cnn_score = len(names & _CNN_INDICATORS)

    if transformer_score > 0 and transformer_score >= cnn_score:
        return "transformer"
    if cnn_score > 0:
        return "cnn"
    return "unknown"


def find_target_layer(model: nn.Module, model_type: str) -> nn.Module | None:
    """
    Attempt to locate a sensible target layer for GradCAM.

    Heuristic: the last Conv2d-containing sequential block before the
    classification head.
    """
    if model_type == "transformer":
        return None  # not needed for attention rollout

    # Walk named modules in reverse; pick last Conv2d
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv
