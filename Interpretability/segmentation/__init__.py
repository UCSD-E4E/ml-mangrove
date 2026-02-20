"""
seg_interpret
=============
Validated interpretation toolkit for PyTorch segmentation models.

Supports CNN-based (GradCAM, GradCAM++) and Transformer-based
(Attention Rollout) architectures, plus model-agnostic LIME and SHAP.

Quick start
-----------
>>> from seg_interpret import SegmentationInterpreter
>>> interp = SegmentationInterpreter(my_model)           # auto-detect type
>>> result = interp.interpret("photo.png", class_idx=1)
>>> result.show()
>>> result.save("explanation.png")

>>> # Run all applicable methods and compare
>>> results = interp.interpret_all("photo.png", class_idx=1)
>>> from seg_interpret import compare_methods
>>> compare_methods(results, image="photo.png")
"""

from .core import SegmentationInterpreter
from .visualization import InterpretationResult, compare_methods, COLORMAPS
from .detect import detect_model_type
from .validation import (
    SegInterpretError,
    InvalidImageError,
    InvalidModelError,
    InvalidMethodError,
    InvalidClassIndexError,
)

__all__ = [
    # Core
    "SegmentationInterpreter",
    # Result & visualisation
    "InterpretationResult",
    "compare_methods",
    "COLORMAPS",
    # Utilities
    "detect_model_type",
    # Exceptions
    "SegInterpretError",
    "InvalidImageError",
    "InvalidModelError",
    "InvalidMethodError",
    "InvalidClassIndexError",
]

__version__ = "0.1.0"
