"""
Sample a fraction of training chips and copy the full val split from a completed
region into a compact replay buffer.

Usage:
  python scripts/create_replay_buffer.py \\
      --chips-dir /data/chips \\
      --region florida \\
      --replay-dir /data/replay \\
      --fraction 0.05

Outputs:
  /data/replay/<region>/train/  — 5% random sample of train chips
  /data/replay/<region>/val/    — full val split (kept for BWT / forgetting eval)

Run immediately after training a region, before deleting the full chip directory.
"""

import argparse
import json
import random
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build a compact replay buffer by sampling chips from a trained region.'
    )
    parser.add_argument('--chips-dir',  required=True, help='Root chip directory (has <region>/train/)')
    parser.add_argument('--region',     required=True, help='Region name to sample from')
    parser.add_argument('--replay-dir', default='/data/replay', help='Root replay directory (default: /data/replay)')
    parser.add_argument('--fraction',   type=float, default=0.05,
                        help='Fraction of chips to keep (default: 0.05 = 5%%)')
    parser.add_argument('--count',      type=int, default=None,
                        help='Hard cap: keep at most this many chips (overrides fraction upper bound)')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    src_dir = Path(args.chips_dir) / args.region / 'train'
    dst_dir = Path(args.replay_dir) / args.region / 'train'

    if not src_dir.exists():
        raise FileNotFoundError(f'Source chip dir not found: {src_dir}')

    all_chips = sorted(src_dir.glob('chip_*.npz'))
    if not all_chips:
        raise FileNotFoundError(f'No chip_*.npz files found in: {src_dir}')

    n_sample = max(1, round(len(all_chips) * args.fraction))
    if args.count is not None:
        n_sample = min(n_sample, args.count)

    selected = random.sample(all_chips, n_sample)
    selected.sort()

    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f'Region       : {args.region}')
    print(f'Source chips : {len(all_chips):,}  ({src_dir})')
    print(f'Replay chips : {n_sample:,}  ({args.fraction:.0%} fraction)')
    print(f'Destination  : {dst_dir}')

    catalog = []
    for i, src_path in enumerate(selected):
        dst_path = dst_dir / f'chip_{i:05d}.npz'
        if dst_path.exists():
            dst_path.unlink()
        shutil.copy2(src_path, dst_path)
        catalog.append({'original': str(src_path), 'replay': str(dst_path)})

    cat_path = dst_dir / 'catalog.json'
    with open(cat_path, 'w') as f:
        json.dump(catalog, f, indent=2)

    print(f'Done. Wrote {n_sample:,} chips → {dst_dir}')
    print(f'Train replay size : {sum(p.stat().st_size for p in dst_dir.glob("chip_*.npz")) / 1e6:.1f} MB')

    # ── Copy full val split ───────────────────────────────────────────────────
    val_src = Path(args.chips_dir) / args.region / 'val'
    val_dst = Path(args.replay_dir) / args.region / 'val'

    if val_src.exists():
        val_chips = sorted(val_src.glob('chip_*.npz'))
        val_dst.mkdir(parents=True, exist_ok=True)
        print(f'\nCopying full val split: {len(val_chips):,} chips → {val_dst}')
        for src_path in val_chips:
            dst_path = val_dst / src_path.name
            if dst_path.exists():
                dst_path.unlink()
            shutil.copy2(src_path, dst_path)
        # copy catalog if present
        val_cat = val_src / 'catalog.json'
        if val_cat.exists():
            shutil.copy2(val_cat, val_dst / 'catalog.json')
        print(f'Val split size    : {sum(p.stat().st_size for p in val_dst.glob("chip_*.npz")) / 1e6:.1f} MB')
    else:
        print(f'WARNING: val dir not found, skipping: {val_src}')


if __name__ == '__main__':
    main()
