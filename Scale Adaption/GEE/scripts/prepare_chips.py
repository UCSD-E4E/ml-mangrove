"""
Download GEE tiles from GCS and chip them into NPZ files for training.
Run on the GCP VM after GEE export tasks have completed.

Usage:
  # Download + chip a single region
  python prepare_chips.py \\
      --bucket e4e-mangrove-tiles \\
      --region brazil \\
      --tiles-dir /data/tiles \\
      --chips-dir /data/chips

  # Chip only (tiles already downloaded)
  python prepare_chips.py \\
      --region brazil \\
      --tiles-dir /data/tiles \\
      --chips-dir /data/chips \\
      --skip-download

  # Verify norm stats only (no chipping)
  python prepare_chips.py --region florida --tiles-dir /data/tiles --stats-only
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import yaml

# ── path setup ────────────────────────────────────────────────────────────────
_GEE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_GEE_ROOT))

from gee_dataset import (   # noqa: E402
    BAND_GROUPS,
    GEETileDataset,
    chip_tiles,
    compute_norm_stats,
)

CONFIG_PATH   = _GEE_ROOT / 'config' / 'regions.yaml'
TAXONOMY_PATH = _GEE_ROOT / 'config' / 'taxonomy.yaml'


def load_taxonomy() -> dict:
    with open(TAXONOMY_PATH) as f:
        t = yaml.safe_load(f)
    return {
        'esa_to_class': {int(k): int(v) for k, v in t['esa_to_class'].items()},
        'num_classes':  t['num_classes'],
        'ignore_index': t['ignore_index'],
    }


def download_tiles(bucket: str, gcs_prefix: str, out_dir: Path) -> None:
    """gsutil rsync GCS prefix → local directory (skips existing files)."""
    from google.cloud import storage

    out_dir.mkdir(parents=True, exist_ok=True)
    client  = storage.Client()
    bkt     = client.bucket(bucket)
    blobs   = list(bkt.list_blobs(prefix=f'{gcs_prefix}/'))
    tifs    = [b for b in blobs if b.name.endswith('.tif')]

    print(f'  {len(tifs)} .tif files in gs://{bucket}/{gcs_prefix}/')
    for blob in tifs:
        fname = Path(blob.name).name
        dest  = out_dir / fname
        if dest.exists():
            print(f'  skip (exists): {fname}')
            continue
        print(f'  downloading  : {fname}', end=' ', flush=True)
        blob.download_to_filename(str(dest))
        print(f'({dest.stat().st_size / 1e6:.1f} MB)')


def get_or_compute_norm_stats(
    tile_paths: list,
    stats_path: Path,
    mode: str,
) -> dict:
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        print(f'  Loaded cached norm stats: {stats_path}')
        return stats

    print('  Computing norm stats (runs once)...')
    mean, std = compute_norm_stats(tile_paths, BAND_GROUPS[mode], sample_stride=16)
    stats = {'mean': mean.tolist(), 'std': std.tolist()}
    with open(stats_path, 'w') as f:
        json.dump(stats, f)
    print(f'  Saved: {stats_path}')
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description='Download GCS tiles and chip to NPZ.')
    parser.add_argument('--bucket',        default=None, help='GCS bucket name')
    parser.add_argument('--region',        required=True)
    parser.add_argument('--tiles-dir',     required=True, help='Local dir for raw .tif files')
    parser.add_argument('--chips-dir',     required=True, help='Local dir for .npz chips')
    parser.add_argument('--mode',          default='full',
                        choices=['rgb', 'rgbn', 'embeddings', 'full'])
    parser.add_argument('--patch-size',    type=int, default=512)
    parser.add_argument('--stride',        type=int, default=256,
                        help='Sliding window stride (< patch-size = overlap)')
    parser.add_argument('--val-split',     type=float, default=0.2)
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument('--skip-download', action='store_true')
    parser.add_argument('--stats-only',    action='store_true',
                        help='Compute/print norm stats only, no chipping')
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        region_cfg = yaml.safe_load(f)[args.region]

    taxonomy = load_taxonomy()
    tiles_dir = Path(args.tiles_dir) / args.region
    chips_dir = Path(args.chips_dir) / args.region

    # ── Download ──────────────────────────────────────────────────────────────
    if not args.skip_download and not args.stats_only:
        if not args.bucket:
            parser.error('--bucket required unless --skip-download is set')
        print(f'\n=== Downloading tiles: {args.region} ===')
        download_tiles(args.bucket, region_cfg['gcs_prefix'], tiles_dir)

    tile_paths = sorted(tiles_dir.glob('*.tif'))
    if not tile_paths:
        print(f'No .tif files found in {tiles_dir}')
        sys.exit(1)
    print(f'\nFound {len(tile_paths)} tiles in {tiles_dir}')

    # ── Norm stats ────────────────────────────────────────────────────────────
    print(f'\n=== Norm stats: {args.region} ===')
    stats_path = tiles_dir / f'{args.region}_norm_stats.json'
    norm_stats = get_or_compute_norm_stats(tile_paths, stats_path, args.mode)

    if args.stats_only:
        return

    # ── Train / val split ─────────────────────────────────────────────────────
    random.seed(args.seed)
    shuffled = list(tile_paths)
    random.shuffle(shuffled)
    n_val        = max(1, int(len(shuffled) * args.val_split))
    val_tiles    = shuffled[:n_val]
    train_tiles  = shuffled[n_val:]
    print(f'\nTiles — train: {len(train_tiles)}, val: {len(val_tiles)}')

    # ── Build datasets and chip ───────────────────────────────────────────────
    print(f'\n=== Chipping: {args.region} ===')
    print('Building train dataset...')
    train_ds = GEETileDataset(
        tile_paths      = train_tiles,
        mode            = args.mode,
        patch_size      = args.patch_size,
        tile_stride     = args.stride,
        label_map       = taxonomy['esa_to_class'],
        norm_stats      = norm_stats,
        augment         = False,   # augmentation applied at train time by CachedChipDataset
        min_valid_ratio = 0.3,
        ignore_index    = taxonomy['ignore_index'],
    )
    print('Building val dataset...')
    val_ds = GEETileDataset(
        tile_paths      = val_tiles,
        mode            = args.mode,
        patch_size      = args.patch_size,
        tile_stride     = args.patch_size,   # no overlap in val
        label_map       = taxonomy['esa_to_class'],
        norm_stats      = norm_stats,
        augment         = False,
        min_valid_ratio = 0.3,
        ignore_index    = taxonomy['ignore_index'],
    )

    chips_dir.mkdir(parents=True, exist_ok=True)
    chip_tiles(train_ds, str(chips_dir), 'train')
    chip_tiles(val_ds,   str(chips_dir), 'val')

    in_ch = len(BAND_GROUPS[args.mode])
    print(f'\nDone. Chips at: {chips_dir}')
    print(f'  Input channels : {in_ch}  (mode={args.mode})')
    print(f'  Num classes    : {taxonomy["num_classes"]}')
    print(f'\nNext step:')
    print(f'  python train.py --chips-dir {args.chips_dir} --region {args.region} ...')


if __name__ == '__main__':
    main()
