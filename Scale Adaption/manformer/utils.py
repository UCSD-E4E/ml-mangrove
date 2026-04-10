"""
utils.py — Manformer class definitions, normalization stats, and colormap.

Class taxonomy matches segformer_training.ipynb exactly (6 classes, no background).
ESA values absent from ESA_TO_CLASS are assigned IGNORE_INDEX (255) during training.
"""

import numpy as np

# ── NAIP Normalization (uint8 0-255, 4-band RGBN) ────────────────────────────
# Starting values adapted from Chesapeake Bay NAIP (Li et al., CVPR 2024).
# Florida NAIP is spectrally similar; recompute from downloaded tiles if needed:
#
#   import rasterio, numpy as np
#   means = []
#   for fn in naip_fns:
#       with rasterio.open(fn) as f:
#           data = f.read().astype(np.float32)       # (4, H, W)
#           means.append(data.mean(axis=(1, 2)))
#   IMAGE_MEANS = np.mean(means, axis=0)
#   IMAGE_STDS  = np.std(means,  axis=0)
#
IMAGE_MEANS = np.array([117.67, 130.39, 121.52, 162.92], dtype=np.float32)
IMAGE_STDS  = np.array([ 39.25,  37.82,  24.24,  60.03], dtype=np.float32)

# ── Class taxonomy ────────────────────────────────────────────────────────────
# Source: ESA WorldCover v200 (band 69 of Florida GeoTIFFs).
# Only these 6 ESA values appear in the Florida coastal dataset.
# Cropland (40), Bare/Sparse (60), Snow/Ice (70), Moss/Lichen (100) are absent
# and are silently mapped to IGNORE_INDEX.
ESA_TO_CLASS = {
    10: 0,   # Tree cover
    30: 1,   # Grassland
    50: 2,   # Built-up  (human infrastructure — buildings, roads)
    80: 3,   # Permanent water bodies
    90: 4,   # Herbaceous wetland
    95: 5,   # Mangroves
}
CLASS_NAMES = [
    'Tree Cover',   # 0  (ESA 10)
    'Grassland',    # 1  (ESA 30)
    'Built-up',     # 2  (ESA 50)
    'Water',        # 3  (ESA 80)
    'Wetland',      # 4  (ESA 90)
    'Mangrove',     # 5  (ESA 95)
]
NUM_CLASSES  = len(CLASS_NAMES)   # 6
IGNORE_INDEX = 255                # pixels with unknown ESA values are ignored

# ── Class colors (float [0, 1] RGB, consistent with segformer_training.ipynb) ─
CLASS_COLORS = np.array([
    [0.05, 0.40, 0.05],   # 0  Tree Cover  — dark green
    [0.70, 0.80, 0.20],   # 1  Grassland   — yellow-green
    [0.90, 0.20, 0.20],   # 2  Built-up    — red
    [0.10, 0.40, 0.95],   # 3  Water       — blue
    [0.50, 0.80, 0.40],   # 4  Wetland     — light green
    [0.05, 0.75, 0.45],   # 5  Mangrove    — teal
], dtype=np.float32)

# ── Vectorized ESA value → class index lookup table ──────────────────────────
# Usage:  remapped = LABEL_CLASS_TO_IDX_MAP[raw_esa_array]
# Any ESA value not in ESA_TO_CLASS maps to IGNORE_INDEX (255).
# Array length is 256 to safely handle any uint8 input.
def _build_label_map() -> np.ndarray:
    lut = np.full(256, fill_value=IGNORE_INDEX, dtype=np.int64)
    for esa_val, cls_idx in ESA_TO_CLASS.items():
        lut[esa_val] = cls_idx
    return lut

LABEL_CLASS_TO_IDX_MAP = _build_label_map()

# ── Colormap for GeoTIFF output (rasterio write_colormap format) ─────────────
# Maps class index → (R, G, B, A) uint8 tuples.
LABEL_IDX_COLORMAP = {
    idx: tuple((CLASS_COLORS[idx] * 255).astype(np.uint8).tolist()) + (255,)
    for idx in range(NUM_CLASSES)
}
LABEL_IDX_COLORMAP[IGNORE_INDEX] = (0, 0, 0, 0)   # transparent for ignored pixels


def make_label_rgb(label_2d: np.ndarray) -> np.ndarray:
    """
    Convert a 2-D class-index map to a displayable float32 RGB image.

    Args:
        label_2d: (H, W) int array of class indices (0–NUM_CLASSES-1 or IGNORE_INDEX).

    Returns:
        (H, W, 3) float32 array in [0, 1].
    """
    h, w = label_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for i, color in enumerate(CLASS_COLORS):
        rgb[label_2d == i] = color
    rgb[label_2d == IGNORE_INDEX] = [0.15, 0.15, 0.15]   # dark grey for ignored pixels
    return rgb
