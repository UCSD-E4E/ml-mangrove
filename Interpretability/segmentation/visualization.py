"""
visualization.py — Heatmap rendering and overlay utilities for seg_interpret.

Produces:
  • Single heatmap (colour-mapped)
  • Overlay of heatmap on original image
  • Side-by-side comparison panel (image | heatmap | overlay)
  • Contour-overlay (heatmap iso-contours drawn on image)
  • Multi-method comparison grid
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive; must come before pyplot import
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image


def _open_file(path: Path) -> None:
    """Open *path* with the OS default viewer (non-blocking)."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", str(path)])
        elif sys.platform == "win32":
            os.startfile(str(path))
    except Exception:
        pass  # viewer unavailable — user can open manually


# ──────────────────────────────────────────────────────────────
# Colour maps available to the user
# ──────────────────────────────────────────────────────────────
_RED_HEAT = LinearSegmentedColormap.from_list("red_heat", ["white", "red"])
matplotlib.colormaps.register(_RED_HEAT, name="red_heat", force=True)

COLORMAPS = {
    "red_heat": _RED_HEAT,
    "jet":      cm.jet,
    "hot":      cm.hot,
    "plasma":   cm.plasma,
    "viridis":  cm.viridis,
    "inferno":  cm.inferno,
    "turbo":    cm.turbo,
    "coolwarm": cm.coolwarm,
}

DEFAULT_COLORMAP = "red_heat"
DEFAULT_ALPHA = 0.6


# ──────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────

def _load_rgb(image_source) -> np.ndarray:
    """
    Accept a file path, PIL Image, or (H, W, 3) numpy array;
    return a (H, W, 3) float32 array in [0, 1].
    """
    if isinstance(image_source, (str, Path)):
        img = Image.open(image_source).convert("RGB")
        return np.array(img, dtype=np.float32) / 255.0
    if isinstance(image_source, Image.Image):
        return np.array(image_source.convert("RGB"), dtype=np.float32) / 255.0
    if isinstance(image_source, np.ndarray):
        img = image_source.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
        return img
    raise TypeError(f"Unsupported image type: {type(image_source)}")


def _apply_colormap(heatmap: np.ndarray, colormap: str = DEFAULT_COLORMAP) -> np.ndarray:
    """
    Apply a matplotlib colormap to a (H, W) float32 array in [0, 1].
    Returns (H, W, 3) float32 RGB array.
    """
    cmap = COLORMAPS.get(colormap, COLORMAPS[DEFAULT_COLORMAP])
    colored = cmap(heatmap)[:, :, :3]   # drop alpha channel
    return colored.astype(np.float32)


def _overlay(
    image_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    alpha: float,
    heatmap_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Alpha-blend the heatmap over the image.

    If *heatmap_weights* (H, W) is provided, per-pixel alpha = weights * alpha,
    so only regions with high activation are tinted and the rest stay transparent.
    """
    if heatmap_weights is not None:
        per_pixel = (heatmap_weights[..., np.newaxis] * alpha).astype(np.float32)
        return np.clip(image_rgb * (1 - per_pixel) + heatmap_rgb * per_pixel, 0, 1)
    return np.clip(image_rgb * (1 - alpha) + heatmap_rgb * alpha, 0, 1)


def _resize_heatmap(heatmap: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a (H, W) heatmap to (target_H, target_W)."""
    import torch
    import torch.nn.functional as F
    h_t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    h_t = F.interpolate(h_t, size=target_shape, mode="bilinear", align_corners=False)
    return h_t.squeeze().numpy().astype(np.float32)


# ──────────────────────────────────────────────────────────────
# InterpretationResult — returned by SegmentationInterpreter
# ──────────────────────────────────────────────────────────────

class InterpretationResult:
    """
    Container returned by :class:`SegmentationInterpreter`.

    Attributes
    ----------
    heatmap : np.ndarray
        Raw normalised heatmap (H, W), float32 in [0, 1].
    method : str
        Name of the method that produced this result.
    class_idx : int
        Explained class index.
    model_type : str
        Detected/specified model type ('cnn' or 'transformer').
    image_source : path / array / None
        The original image for overlay generation.
    """

    def __init__(
        self,
        heatmap: np.ndarray,
        method: str,
        class_idx: int,
        model_type: str,
        image_source=None,
        class_name: Optional[str] = None,
    ) -> None:
        self.heatmap = heatmap
        self.method = method
        self.class_idx = class_idx
        self.class_name = class_name
        self.model_type = model_type
        self._image_source = image_source

    def _class_label(self) -> str:
        """Return 'ClassName (idx)' if class_name is set, else just 'class idx'."""
        if self.class_name:
            return f"{self.class_name} ({self.class_idx})"
        return f"class {self.class_idx}"

    # ── Rendering ──────────────────────────────────────────────

    def render_heatmap(
        self,
        colormap: str = DEFAULT_COLORMAP,
    ) -> np.ndarray:
        """Return (H, W, 3) float32 RGB of the colour-mapped heatmap."""
        return _apply_colormap(self.heatmap, colormap)

    def render_overlay(
        self,
        image=None,
        alpha: float = DEFAULT_ALPHA,
        colormap: str = DEFAULT_COLORMAP,
    ) -> np.ndarray:
        """
        Return (H, W, 3) float32 RGB of the heatmap blended over the image.

        If *image* is None, uses the image_source provided at construction.
        """
        src = image or self._image_source
        if src is None:
            raise ValueError("No image provided for overlay. Pass image= or set image_source.")
        img_np = _load_rgb(src)
        hm = _resize_heatmap(self.heatmap, (img_np.shape[0], img_np.shape[1]))
        hm_rgb = _apply_colormap(hm, colormap)
        return _overlay(img_np, hm_rgb, alpha, heatmap_weights=hm)

    def render_contour_overlay(
        self,
        image=None,
        n_levels: int = 6,
        colormap: str = DEFAULT_COLORMAP,
    ) -> np.ndarray:
        """
        Draw contour lines of the heatmap on top of the image.
        Returns (H, W, 3) float32 RGB.
        """
        src = image or self._image_source
        if src is None:
            raise ValueError("No image provided. Pass image= or set image_source.")
        img_np = _load_rgb(src)
        H, W = img_np.shape[:2]
        hm = _resize_heatmap(self.heatmap, (H, W))

        fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
        ax.imshow(img_np)
        ax.contour(hm, levels=n_levels, cmap=colormap, alpha=0.8)
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        rendered = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        rendered = rendered.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return rendered.astype(np.float32) / 255.0

    # ── Display helpers ─────────────────────────────────────────

    def show(
        self,
        image=None,
        alpha: float = DEFAULT_ALPHA,
        colormap: str = DEFAULT_COLORMAP,
        figsize: Tuple[int, int] = (14, 4),
    ) -> None:
        """
        Display a three-panel figure:
        [Original image | Heatmap | Overlay]
        """
        src = image or self._image_source
        titles = [
            f"Original",
            f"{self.method}  ({self._class_label()})",
            "Overlay",
        ]
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        if src is not None:
            img_np = _load_rgb(src)
            axes[0].imshow(img_np)
            overlay = self.render_overlay(src, alpha, colormap)
            axes[2].imshow(overlay)
        else:
            axes[0].text(0.5, 0.5, "No image", ha="center", va="center")
            axes[2].text(0.5, 0.5, "No image", ha="center", va="center")

        axes[1].imshow(self.heatmap, cmap=colormap, vmin=0, vmax=1)
        fig.colorbar(
            cm.ScalarMappable(cmap=colormap),
            ax=axes[1], fraction=0.046, pad=0.04,
        )

        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=11)
            ax.axis("off")

        fig.suptitle(
            f"Interpretation  |  model_type={self.model_type}  |  method={self.method}",
            fontsize=12, y=1.01,
        )
        plt.tight_layout()
        tmp = Path(tempfile.mktemp(suffix=".png"))
        fig.savefig(tmp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _open_file(tmp)

    def save(
        self,
        path: str | Path,
        image=None,
        alpha: float = DEFAULT_ALPHA,
        colormap: str = DEFAULT_COLORMAP,
        figsize: Tuple[int, int] = (14, 4),
        dpi: int = 150,
    ) -> Path:
        """
        Save the three-panel figure to *path*.

        Returns
        -------
        Path to the saved file.
        """
        path = Path(path)
        src = image or self._image_source

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        if src is not None:
            img_np = _load_rgb(src)
            axes[0].imshow(img_np)
            overlay = self.render_overlay(src, alpha, colormap)
            axes[2].imshow(overlay)
        else:
            axes[0].text(0.5, 0.5, "No image", ha="center", va="center")
            axes[2].text(0.5, 0.5, "No image", ha="center", va="center")

        axes[1].imshow(self.heatmap, cmap=colormap, vmin=0, vmax=1)
        fig.colorbar(
            cm.ScalarMappable(cmap=colormap),
            ax=axes[1], fraction=0.046, pad=0.04,
        )

        titles = ["Original", f"{self.method}  ({self._class_label()})", "Overlay"]
        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=11)
            ax.axis("off")

        fig.suptitle(
            f"Interpretation  |  model_type={self.model_type}  |  method={self.method}",
            fontsize=12, y=1.01,
        )
        plt.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def __repr__(self) -> str:
        H, W = self.heatmap.shape
        cls = f"{self.class_name!r} ({self.class_idx})" if self.class_name else str(self.class_idx)
        return (
            f"InterpretationResult("
            f"method={self.method!r}, class={cls}, "
            f"model_type={self.model_type!r}, heatmap=({H}×{W}))"
        )


# ──────────────────────────────────────────────────────────────
# Multi-method comparison
# ──────────────────────────────────────────────────────────────

def compare_methods(
    results: Dict[str, InterpretationResult],
    image=None,
    alpha: float = DEFAULT_ALPHA,
    colormap: str = DEFAULT_COLORMAP,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str | Path] = None,
    dpi: int = 150,
) -> None:
    """
    Display (or save) a comparison grid of multiple InterpretationResults.

    Parameters
    ----------
    results : dict mapping method name → InterpretationResult
    image : optional image override (path, PIL, ndarray)
    alpha : blend factor for overlays
    colormap : matplotlib colormap name
    figsize : figure size; auto-computed if None
    save_path : if given, save instead of showing
    dpi : dots per inch for saved figure
    """
    n = len(results)
    if n == 0:
        raise ValueError("No results to compare.")

    # Each method: 2 columns (heatmap + overlay)
    ncols = n * 2
    nrows = 1
    if figsize is None:
        figsize = (ncols * 3 + 1, 4)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    for col_pair, (method_name, res) in enumerate(results.items()):
        src = image or res._image_source
        ax_hm = axes[col_pair * 2]
        ax_ov = axes[col_pair * 2 + 1]

        ax_hm.imshow(res.heatmap, cmap=colormap, vmin=0, vmax=1)
        ax_hm.set_title(f"{method_name}\nheatmap", fontsize=9)
        ax_hm.axis("off")

        if src is not None:
            overlay = res.render_overlay(src, alpha, colormap)
            ax_ov.imshow(overlay)
        else:
            ax_ov.text(0.5, 0.5, "No image", ha="center", va="center")
        ax_ov.set_title(f"{method_name}\noverlay", fontsize=9)
        ax_ov.axis("off")

    fig.suptitle(f"Method Comparison  |  class_idx={list(results.values())[0].class_idx}", fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        tmp = Path(tempfile.mktemp(suffix=".png"))
        fig.savefig(tmp, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        _open_file(tmp)
