"""
core.py — SegmentationInterpreter: the primary public-facing class.

Usage
-----
    from seg_interpret import SegmentationInterpreter

    interp = SegmentationInterpreter(model)                         # auto-detect
    interp = SegmentationInterpreter(model, model_type="cnn")       # explicit
    interp = SegmentationInterpreter(model, model_type="transformer")

    result = interp.interpret("image.png", class_idx=1)             # method="auto"
    result = interp.interpret("image.png", class_idx=1, method="gradcam++")

    result.show()                   # three-panel display
    result.save("out.png")          # save to disk
    result.heatmap                  # raw (H, W) float32 ndarray

    # Compare all applicable methods:
    results = interp.interpret_all("image.png", class_idx=1)
    from seg_interpret import compare_methods
    compare_methods(results, image="image.png")
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .detect import detect_model_type, find_target_layer
from .validation import (
    validate_image_path,
    validate_image_tensor,
    validate_method,
    validate_model,
    validate_model_output,
    validate_class_index,
    InvalidMethodError,
)
from .visualization import InterpretationResult


# ──────────────────────────────────────────────────────────────
# Default image pre-processing pipeline
# ──────────────────────────────────────────────────────────────
_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _load_image_tensor(
    image_path: Union[str, Path],
    transform,
    device: torch.device,
) -> torch.Tensor:
    """Load an image from *image_path* and return a (1, C, H, W) tensor."""
    path = validate_image_path(image_path)
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    validate_image_tensor(tensor)
    return tensor


def _unwrap_output(out) -> torch.Tensor:
    """
    Extract a raw tensor from a model output that may be:
      - a plain torch.Tensor
      - a HuggingFace ModelOutput (e.g. SemanticSegmenterOutput)
      - a dict with common keys
      - a tuple/list (first element taken)
    """
    if isinstance(out, torch.Tensor):
        return out
    # HuggingFace ModelOutput objects expose logits or last_hidden_state
    for attr in ("logits", "last_hidden_state", "pred_masks", "masks_queries_logits"):
        if hasattr(out, attr):
            val = getattr(out, attr)
            if isinstance(val, torch.Tensor):
                return val
    if isinstance(out, dict):
        for key in ("logits", "last_hidden_state", "out", "output"):
            if key in out and isinstance(out[key], torch.Tensor):
                return out[key]
        # fallback: first tensor value
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v
    if isinstance(out, (list, tuple)):
        for item in out:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(
        f"Cannot extract a tensor from model output of type {type(out).__name__}. "
        "Consider wrapping your model's forward() to return raw logits."
    )


def _infer_num_classes(model: nn.Module, input_tensor: torch.Tensor) -> int:
    """Run a quick forward pass to determine the number of output classes."""
    with torch.no_grad():
        out = _unwrap_output(model(input_tensor))
    if out.ndim == 4:   # (N, C, H, W)
        return out.shape[1]
    if out.ndim == 3:   # (N, H, W) — single-channel
        return int(out.max().item()) + 1
    return 1


# ──────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────

class SegmentationInterpreter:
    """
    Unified interpretation interface for segmentation models.

    Parameters
    ----------
    model : nn.Module
        PyTorch segmentation model.  Should be in eval mode or will be set
        to eval mode automatically.
    model_type : 'auto' | 'cnn' | 'transformer'
        Architecture family.  'auto' (default) heuristically detects the type.
    target_layer : nn.Module, optional
        For GradCAM/GradCAM++: the Conv2d layer to hook.  Auto-detected if
        not provided.
    transform : callable, optional
        Image pre-processing pipeline.  Must accept a PIL.Image and return
        a (C, H, W) tensor.  Defaults to standard ImageNet normalisation at
        512 × 512.
    device : torch.device or str, optional
        Inference device.  Defaults to CUDA if available, else CPU.
    num_classes : int, optional
        Number of output classes.  Auto-inferred from a forward pass if not
        specified.
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str = "auto",
        target_layer: Optional[nn.Module] = None,
        transform=None,
        device: Optional[Union[str, torch.device]] = None,
        num_classes: Optional[int] = None,
    ) -> None:
        validate_model(model)

        # Device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.model = model.to(self.device).eval()
        self.transform = transform or _DEFAULT_TRANSFORM

        # Detect model type
        if model_type == "auto":
            self.model_type = detect_model_type(model)
            if self.model_type == "unknown":
                warnings.warn(
                    "Could not reliably detect model type. "
                    "Defaulting to 'cnn'.  Specify model_type='transformer' if needed."
                )
                self.model_type = "cnn"
        else:
            if model_type not in ("cnn", "transformer"):
                raise ValueError(f"model_type must be 'auto', 'cnn', or 'transformer'; got {model_type!r}")
            self.model_type = model_type

        # Target layer for GradCAM
        if target_layer is not None:
            self.target_layer = target_layer
        else:
            self.target_layer = find_target_layer(model, self.model_type)
            if self.target_layer is None and self.model_type == "cnn":
                warnings.warn(
                    "Could not auto-detect a target Conv2d layer for GradCAM. "
                    "Provide target_layer= explicitly if you want gradient-based methods."
                )

        # Num classes (lazy init on first interpret call if None)
        self._num_classes = num_classes

    # ── Private helpers ─────────────────────────────────────────

    def _get_num_classes(self, input_tensor: torch.Tensor) -> int:
        if self._num_classes is None:
            self._num_classes = _infer_num_classes(self.model, input_tensor)
        return self._num_classes

    def _resolve_class_idx(self, input_tensor: torch.Tensor, class_idx: Optional[int]) -> int:
        n = self._get_num_classes(input_tensor)
        if class_idx is None:
            with torch.no_grad():
                out = _unwrap_output(self.model(input_tensor))
            if out.ndim == 4:
                class_idx = int(out.squeeze(0).mean(dim=(1, 2)).argmax().item())
            else:
                class_idx = 0
        validate_class_index(class_idx, n)
        return class_idx

    def _auto_method(self) -> str:
        """Default to Integrated Gradients — fast, smooth, model-agnostic."""
        return "integrated_gradients"

    # ── Public API ──────────────────────────────────────────────

    def interpret(
        self,
        image_path: Union[str, Path],
        class_idx: Optional[int] = None,
        method: str = "auto",
        smooth_sigma: float = 4.0,
        class_name: Optional[str] = None,
    ) -> InterpretationResult:
        """
        Interpret the segmentation model for a single class on one image.

        Parameters
        ----------
        image_path : str or Path
            Path to the input image (PNG/JPG).
        class_idx : int, optional
            The class to explain.  Defaults to the model's dominant class.
        method : str
            One of 'auto', 'gradcam', 'gradcam++', 'attention_rollout',
            'lime', 'shap'.  'auto' selects the best method for the
            detected model type.
        smooth_sigma : float
            Gaussian blur radius applied to LIME/SHAP heatmaps to produce
            smooth gradients instead of hard superpixel blocks.
            0 disables smoothing.

        Returns
        -------
        :class:`~seg_interpret.visualization.InterpretationResult`
        """
        method = validate_method(method)
        if method == "auto":
            method = self._auto_method()

        input_tensor = _load_image_tensor(image_path, self.transform, self.device)
        class_idx = self._resolve_class_idx(input_tensor, class_idx)

        # Validate output shape
        with torch.no_grad():
            out = _unwrap_output(self.model(input_tensor))
        validate_model_output(out, self._num_classes, tuple(input_tensor.shape))

        output_size = (input_tensor.shape[-2], input_tensor.shape[-1])

        # ── Dispatch ─────────────────────────────────────────────
        if method == "gradcam":
            heatmap = self._run_gradcam(input_tensor, class_idx, output_size, plus_plus=False)
        elif method == "gradcam++":
            heatmap = self._run_gradcam(input_tensor, class_idx, output_size, plus_plus=True)
        elif method == "attention_rollout":
            heatmap = self._run_attention_rollout(input_tensor, class_idx, output_size)
        elif method == "integrated_gradients":
            heatmap = self._run_integrated_gradients(input_tensor, class_idx, output_size, smooth_sigma)
        elif method == "lime":
            heatmap = self._run_lime(input_tensor, class_idx, output_size, smooth_sigma)
        elif method == "shap":
            heatmap = self._run_shap(input_tensor, class_idx, output_size, smooth_sigma)
        else:
            raise InvalidMethodError(f"Unhandled method: {method!r}")

        return InterpretationResult(
            heatmap=heatmap,
            method=method,
            class_idx=class_idx,
            model_type=self.model_type,
            image_source=image_path,
            class_name=class_name,
        )

    def interpret_all(
        self,
        image_path: Union[str, Path],
        class_idx: Optional[int] = None,
        methods: Optional[list[str]] = None,
    ) -> Dict[str, InterpretationResult]:
        """
        Run multiple interpretation methods and return a dict of results.

        Parameters
        ----------
        image_path : str or Path
        class_idx : int, optional
        methods : list of str, optional
            Subset of methods to run.  Defaults to all methods appropriate
            for the detected model type.

        Returns
        -------
        dict mapping method name → InterpretationResult
        """
        if methods is None:
            methods = ["integrated_gradients", "lime", "shap"]

        results = {}
        for m in methods:
            try:
                results[m] = self.interpret(image_path, class_idx=class_idx, method=m)
            except Exception as exc:
                warnings.warn(f"Method '{m}' failed: {exc}")
        return results

    # ── Method runners ───────────────────────────────────────────

    def _run_gradcam(self, tensor, class_idx, output_size, plus_plus=False):
        if self.target_layer is None:
            raise RuntimeError(
                "No target_layer available for GradCAM. "
                "Provide target_layer= when constructing SegmentationInterpreter."
            )
        from .methods.gradcam import gradcam, gradcam_plus_plus
        fn = gradcam_plus_plus if plus_plus else gradcam
        return fn(self.model, tensor, self.target_layer, class_idx, output_size)

    def _run_attention_rollout(self, tensor, class_idx, output_size):
        from .methods.attention import attention_rollout
        return attention_rollout(self.model, tensor, class_idx, output_size=output_size)

    def _run_integrated_gradients(self, tensor, class_idx, output_size, smooth_sigma=2.0):
        from .methods.integrated_gradients import integrated_gradients
        return integrated_gradients(self.model, tensor, class_idx,
                                    output_size=output_size, smooth_sigma=smooth_sigma)

    def _run_lime(self, tensor, class_idx, output_size, smooth_sigma=8.0):
        from .methods.perturbation import lime_segmentation
        return lime_segmentation(self.model, tensor, class_idx,
                                 output_size=output_size, smooth_sigma=smooth_sigma)

    def _run_shap(self, tensor, class_idx, output_size, smooth_sigma=8.0):
        from .methods.perturbation import shap_segmentation
        return shap_segmentation(self.model, tensor, class_idx,
                                 output_size=output_size, smooth_sigma=smooth_sigma)
