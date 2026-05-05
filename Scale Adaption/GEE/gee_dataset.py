"""
Shared dataset and model classes for the GEE SegFormer pipeline.
Imported by prepare_chips.py, train.py, and optionally the training notebook.
"""

import json
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import rasterio.windows
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

# ── DroneClassification imports ───────────────────────────────────────────────
_DC_ROOT = Path(__file__).parent.parent.parent / 'DroneClassification'
sys.path.insert(0, str(_DC_ROOT))

from models import SegFormer  # noqa: E402

# ── Band layout (0-indexed) ───────────────────────────────────────────────────
BAND_GROUPS = {
    'rgb':        list(range(0, 3)),
    'rgbn':       list(range(0, 4)),
    'embeddings': list(range(4, 68)),
    'full':       list(range(0, 68)),
}
LABEL_BAND = 68   # 0-based index in the 69-band GeoTIFF


# ─────────────────────────────────────────────────────────────────────────────
# Tile dataset
# ─────────────────────────────────────────────────────────────────────────────

class GEETileDataset(Dataset):
    """
    Lazy-loading dataset for GEE 69-band Sentinel-2 GeoTIFFs.

    Sliding window over 2048×2048 tiles with configurable patch size and stride.
    TILE_STRIDE < PATCH_SIZE gives overlapping chips for more training samples.

    Returns: (features [C, H, W] float32, label [H, W] int64)
    """

    def __init__(
        self,
        tile_paths: list,
        mode: str = 'full',
        patch_size: int = 512,
        tile_stride: Optional[int] = None,
        label_map: Optional[dict] = None,
        norm_stats: Optional[dict] = None,
        augment: bool = False,
        min_valid_ratio: float = 0.3,
        ignore_index: int = 255,
    ):
        self.tile_paths    = [str(p) for p in tile_paths]
        self.patch_size    = patch_size
        self.tile_stride   = tile_stride if tile_stride is not None else patch_size
        self.feature_bands = BAND_GROUPS[mode]
        self.label_map     = label_map or {}
        self.augment       = augment
        self.ignore_index  = ignore_index

        if norm_stats is not None:
            all_mean = np.array(norm_stats['mean'], dtype=np.float32)
            all_std  = np.array(norm_stats['std'],  dtype=np.float32)
            self.mean = all_mean[self.feature_bands]
            self.std  = all_std[self.feature_bands]
        else:
            self.mean = None
            self.std  = None

        self.catalog = self._build_catalog(min_valid_ratio)
        print(f'  {len(self.catalog):,} chips ({patch_size}×{patch_size}, '
              f'stride={self.tile_stride}) from {len(tile_paths)} tiles '
              f'(min_valid={min_valid_ratio})')

    def _build_catalog(self, min_valid_ratio: float) -> list:
        known_esa = set(self.label_map.keys())
        catalog = []
        for path in tqdm(self.tile_paths, desc='Cataloguing tiles', leave=False):
            with rasterio.open(path) as src:
                H, W   = src.height, src.width
                nodata = src.nodata
                for row_off in range(0, H - self.patch_size + 1, self.tile_stride):
                    for col_off in range(0, W - self.patch_size + 1, self.tile_stride):
                        win   = rasterio.windows.Window(col_off, row_off, self.patch_size, self.patch_size)
                        label = src.read(LABEL_BAND + 1, window=win)
                        finite = np.isfinite(label)
                        if nodata is not None and np.isfinite(nodata):
                            finite &= (label != nodata)
                        valid = np.isin(label[finite].astype(np.int32), list(known_esa))
                        ratio = valid.sum() / label.size if label.size > 0 else 0.0
                        if ratio >= min_valid_ratio:
                            catalog.append((path, row_off, col_off))
        return catalog

    def __len__(self) -> int:
        return len(self.catalog)

    def __getitem__(self, idx: int):
        path, row_off, col_off = self.catalog[idx]
        win = rasterio.windows.Window(col_off, row_off, self.patch_size, self.patch_size)
        with rasterio.open(path) as src:
            feat_1idx = [b + 1 for b in self.feature_bands]
            features  = src.read(feat_1idx, window=win).astype(np.float32)
            label_f   = src.read(LABEL_BAND + 1, window=win)

        features = np.where(np.isfinite(features), features, 0.0)
        label_f  = np.where(np.isfinite(label_f), label_f, -1.0)
        label    = label_f.astype(np.int64)

        if self.mean is not None:
            features = (features - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-8)

        remapped = np.full_like(label, fill_value=self.ignore_index)
        for esa_val, class_idx in self.label_map.items():
            remapped[label == esa_val] = class_idx
        label = remapped

        if self.augment:
            if random.random() > 0.5:
                features = np.flip(features, axis=2).copy()
                label    = np.flip(label, axis=1).copy()
            if random.random() > 0.5:
                features = np.flip(features, axis=1).copy()
                label    = np.flip(label, axis=0).copy()
            k = random.randint(0, 3)
            if k:
                features = np.rot90(features, k=k, axes=(1, 2)).copy()
                label    = np.rot90(label, k=k, axes=(0, 1)).copy()

        return torch.from_numpy(features), torch.from_numpy(label).long()


# ─────────────────────────────────────────────────────────────────────────────
# Cached chip dataset (loads pre-extracted .npz files — no rasterio at train time)
# ─────────────────────────────────────────────────────────────────────────────

class CachedChipDataset(Dataset):
    """Loads pre-extracted chips from .npz files."""

    def __init__(self, cache_dir: str, augment: bool = False):
        self.files   = sorted(Path(cache_dir).glob('chip_*.npz'))
        self.augment = augment

        cat_path = Path(cache_dir) / 'catalog.json'
        if cat_path.exists():
            with open(cat_path) as f:
                raw = json.load(f)
            try:
                self.catalog = [(d['path'], d['row_off'], d['col_off']) for d in raw]
            except KeyError:
                self.catalog = None  # replay buffer catalog has a different schema
        else:
            self.catalog = None

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        d        = np.load(self.files[idx])
        features = d['features'].astype(np.float32)
        label    = d['label'].astype(np.int64)

        if self.augment:
            if random.random() > 0.5:
                features = np.flip(features, axis=2).copy()
                label    = np.flip(label, axis=1).copy()
            if random.random() > 0.5:
                features = np.flip(features, axis=1).copy()
                label    = np.flip(label, axis=0).copy()
            k = random.randint(0, 3)
            if k:
                features = np.rot90(features, k=k, axes=(1, 2)).copy()
                label    = np.rot90(label, k=k, axes=(0, 1)).copy()

        return torch.from_numpy(features), torch.from_numpy(label).long()


def chip_tiles(
    dataset: GEETileDataset,
    out_dir: str,
    split_name: str,
) -> CachedChipDataset:
    """Write a GEETileDataset to .npz chips under out_dir/split_name/."""
    split_dir = Path(out_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    existing = {f.stem for f in split_dir.glob('chip_*.npz')}
    to_do    = [i for i in range(len(dataset)) if f'chip_{i:05d}' not in existing]

    if not to_do:
        print(f'{split_name}: all {len(dataset):,} chips already cached → {split_dir}')
    else:
        print(f'{split_name}: writing {len(to_do):,} chips → {split_dir}')
        for idx in tqdm(to_do, desc=f'Caching {split_name}'):
            features, label = dataset[idx]
            np.savez(
                split_dir / f'chip_{idx:05d}',
                features=features.numpy().astype(np.float16),
                label=label.numpy().astype(np.uint8),
            )

    cat_path = split_dir / 'catalog.json'
    if not cat_path.exists() and hasattr(dataset, 'catalog'):
        with open(cat_path, 'w') as f:
            json.dump(
                [{'path': str(p), 'row_off': int(r), 'col_off': int(c)}
                 for p, r, c in dataset.catalog],
                f,
            )

    return CachedChipDataset(str(split_dir), augment=(split_name == 'train'))


# ─────────────────────────────────────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────────────────────────────────────

def compute_norm_stats(
    tile_paths: list,
    feature_bands: list,
    sample_stride: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Streaming mean/std via pixel subsampling. Returns (mean, std) float32."""
    n       = len(feature_bands)
    sums    = np.zeros(n, dtype=np.float64)
    sq_sums = np.zeros(n, dtype=np.float64)
    counts  = np.zeros(n, dtype=np.int64)

    feat_1idx = [b + 1 for b in feature_bands]
    for path in tqdm(tile_paths, desc='Computing norm stats'):
        with rasterio.open(path) as src:
            data = src.read(feat_1idx).astype(np.float64)
        data = data[:, ::sample_stride, ::sample_stride].reshape(n, -1)
        for c in range(n):
            valid = data[c][np.isfinite(data[c])]
            sums[c]    += valid.sum()
            sq_sums[c] += (valid ** 2).sum()
            counts[c]  += valid.size

    mean = (sums / counts).astype(np.float32)
    std  = np.sqrt(np.clip(sq_sums / counts - (sums / counts) ** 2, 0, None)).astype(np.float32)
    return mean, std


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class GEESegFormer(nn.Module):
    """
    SegFormer adapted for multi-channel GEE satellite imagery.
    A two-layer 1×1 conv maps N input bands → 3 channels for the pretrained MiT backbone.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        segformer_weights: str = 'nvidia/segformer-b2-finetuned-ade-512-512',
        patch_size: int = 512,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        if in_channels == 3:
            self.input_proj = nn.Identity()
        else:
            hidden = min(64, max(8, in_channels // 2))
            self.input_proj = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.GELU(),
                nn.Conv2d(hidden, 3, kernel_size=1, bias=False),
            )

        self.segformer = SegFormer(
            num_classes=num_classes,
            input_image_size=patch_size,
            weights=segformer_weights,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.segformer(self.input_proj(x))
