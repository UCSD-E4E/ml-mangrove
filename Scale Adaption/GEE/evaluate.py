"""
Comprehensive evaluation for GEE SegFormer — single region, forgetting check, or full CL audit.

Usage examples:

  # After Florida training — evaluate current region
  python evaluate.py \\
      --checkpoint /data/experiments/gee_florida_v1/best_model.pth \\
      --chips-dir /data/chips \\
      --regions florida \\
      --output-dir /data/evaluations/florida

  # After Brazil CL training — check forgetting on Florida + current on Brazil
  python evaluate.py \\
      --checkpoint /data/experiments/gee_brazil_cl_v1/best_model.pth \\
      --chips-dir /data/chips \\
      --regions florida brazil \\
      --baseline /data/experiments/gee_florida_v1/best_model.pth \\
      --output-dir /data/evaluations/brazil_cl_check

  # Full audit across all trained regions
  python evaluate.py \\
      --checkpoint /data/experiments/gee_indonesia_cl_v1/best_model.pth \\
      --chips-dir /data/chips \\
      --regions florida brazil indonesia \\
      --output-dir /data/evaluations/indonesia_full
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

_GEE_ROOT = Path(__file__).parent
sys.path.insert(0, str(_GEE_ROOT))
sys.path.insert(0, str(_GEE_ROOT.parent.parent / 'DroneClassification'))

from gee_dataset import BAND_GROUPS, CachedChipDataset, GEESegFormer  # noqa: E402

TAXONOMY_PATH     = _GEE_ROOT / 'config' / 'taxonomy.yaml'
SEGFORMER_WEIGHTS = 'nvidia/segformer-b4-finetuned-ade-512-512'

CLASS_COLORS = [
    '#2d6a4f',  # Trees
    '#74c69d',  # Shrubland
    '#d9ed92',  # Grassland
    '#f4d35e',  # Cropland
    '#c77dff',  # Built-up
    '#d4a373',  # Bare/Sparse
    '#4895ef',  # Water
    '#90e0ef',  # Wetland
    '#1b4332',  # Mangrove
]
IGNORE_COLOR = '#111111'


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def build_confusion_matrix(
    model: torch.nn.Module,
    loader: DataLoader,
    num_classes: int,
    ignore_index: int,
    device: torch.device,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc='  inference', leave=False):
            x = x.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = model(x)
            preds  = logits.argmax(dim=1).cpu().numpy()
            labels = y.numpy()
            mask   = labels != ignore_index
            np.add.at(cm, (labels[mask], preds[mask]), 1)
    return cm


def metrics_from_cm(cm: np.ndarray, class_names: list[str]) -> dict:
    n = cm.shape[0]
    iou, prec, rec, f1 = [], [], [], []
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = tp + fn
        _iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float('nan')
        _prec = tp / (tp + fp)      if (tp + fp) > 0      else float('nan')
        _rec  = tp / (tp + fn)      if (tp + fn) > 0      else float('nan')
        _f1   = (2 * _prec * _rec / (_prec + _rec)) if (not np.isnan(_prec + _rec) and (_prec + _rec) > 0) else float('nan')
        iou.append(_iou if support > 0 else float('nan'))
        prec.append(_prec)
        rec.append(_rec)
        f1.append(_f1)

    present = [not np.isnan(v) for v in iou]
    miou    = float(np.nanmean(iou))
    macc    = float(cm.diagonal().sum() / cm.sum()) if cm.sum() > 0 else 0.0

    return {
        'miou':            miou,
        'pixel_accuracy':  macc,
        'num_classes_present': sum(present),
        'class_iou':       {n: float(v) for n, v in zip(class_names, iou)},
        'class_precision': {n: float(v) for n, v in zip(class_names, prec)},
        'class_recall':    {n: float(v) for n, v in zip(class_names, rec)},
        'class_f1':        {n: float(v) for n, v in zip(class_names, f1)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix (row-normalized)', fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_class_metrics(metrics: dict, class_names: list[str], region: str, out_path: Path) -> None:
    iou  = [metrics['class_iou'].get(n, float('nan'))       for n in class_names]
    prec = [metrics['class_precision'].get(n, float('nan')) for n in class_names]
    rec  = [metrics['class_recall'].get(n, float('nan'))    for n in class_names]
    f1   = [metrics['class_f1'].get(n, float('nan'))        for n in class_names]

    x    = np.arange(len(class_names))
    w    = 0.2
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 1.5*w, iou,  w, label='IoU',       color='#4361ee')
    ax.bar(x - 0.5*w, prec, w, label='Precision', color='#7209b7')
    ax.bar(x + 0.5*w, rec,  w, label='Recall',    color='#f72585')
    ax.bar(x + 1.5*w, f1,   w, label='F1',        color='#4cc9f0')
    ax.axhline(metrics['miou'], color='#4361ee', linestyle='--', linewidth=1, alpha=0.7, label=f"mIoU={metrics['miou']:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title(f'Per-class metrics — {region}  |  PixAcc={metrics["pixel_accuracy"]:.3f}')
    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_samples(
    dataset: CachedChipDataset,
    model: torch.nn.Module,
    num_classes: int,
    ignore_index: int,
    class_names: list[str],
    device: torch.device,
    out_path: Path,
    n_samples: int = 6,
    has_rgb: bool = True,
) -> None:
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    cmap    = matplotlib.colors.ListedColormap(CLASS_COLORS[:num_classes])
    norm    = matplotlib.colors.BoundaryNorm(range(num_classes + 1), cmap.N)

    n_cols = 3 if has_rgb else 2
    fig, axes = plt.subplots(len(indices), n_cols, figsize=(5 * n_cols, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    with torch.no_grad():
        for row, idx in enumerate(indices):
            feat, label = dataset[idx]
            x = feat.unsqueeze(0).to(device)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = model(x)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

            col = 0
            if has_rgb:
                rgb = feat[:3].numpy().transpose(1, 2, 0)
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
                axes[row, col].imshow(rgb)
                axes[row, col].set_title(f'RGB  (chip {idx})')
                axes[row, col].axis('off')
                col += 1

            gt_display = label.numpy().astype(float)
            gt_display[gt_display == ignore_index] = -1
            axes[row, col].imshow(gt_display, cmap=cmap, norm=norm, interpolation='nearest')
            axes[row, col].set_title('Ground Truth')
            axes[row, col].axis('off')
            col += 1

            axes[row, col].imshow(pred, cmap=cmap, norm=norm, interpolation='nearest')
            axes[row, col].set_title('Prediction')
            axes[row, col].axis('off')

    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=class_names[i]) for i in range(num_classes)]
    fig.legend(handles=patches, loc='lower center', ncol=num_classes, fontsize=8, bbox_to_anchor=(0.5, 0))
    plt.suptitle(f'Sample Predictions — {out_path.stem}', fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_cl_comparison(
    all_metrics: dict[str, dict],
    class_names: list[str],
    out_path: Path,
) -> None:
    regions = list(all_metrics.keys())
    x = np.arange(len(class_names))
    w = 0.8 / len(regions)
    palette = plt.cm.tab10(np.linspace(0, 0.9, len(regions)))

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, (region, metrics) in enumerate(all_metrics.items()):
        iou_vals = [metrics['class_iou'].get(n, float('nan')) for n in class_names]
        ax.bar(x + (i - len(regions)/2 + 0.5) * w, iou_vals, w,
               label=region, color=palette[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('IoU')
    ax.set_title('Per-class IoU across evaluated regions')
    ax.legend()
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_forgetting(
    current_metrics: dict[str, dict],
    baseline_metrics: dict[str, dict],
    class_names: list[str],
    out_path: Path,
) -> None:
    common_regions = [r for r in current_metrics if r in baseline_metrics]
    if not common_regions:
        return

    fig, axes = plt.subplots(1, len(common_regions),
                             figsize=(9 * len(common_regions), 6), squeeze=False)

    for col, region in enumerate(common_regions):
        deltas = []
        for name in class_names:
            cur  = current_metrics[region]['class_iou'].get(name, float('nan'))
            base = baseline_metrics[region]['class_iou'].get(name, float('nan'))
            deltas.append(cur - base if not (np.isnan(cur) or np.isnan(base)) else float('nan'))

        colors = ['#e63946' if d < 0 else '#2a9d8f' for d in
                  [0 if np.isnan(d) else d for d in deltas]]
        ax = axes[0, col]
        ax.barh(class_names, deltas, color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('ΔIoU  (current − baseline)')
        ax.set_title(f'Forgetting on {region}\n(red=forgetting, green=improvement)')

    plt.suptitle('Continual Learning — Catastrophic Forgetting Analysis', fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_miou_summary(
    all_metrics: dict[str, dict],
    out_path: Path,
) -> None:
    regions = list(all_metrics.keys())
    mious   = [all_metrics[r]['miou'] for r in regions]
    paccs   = [all_metrics[r]['pixel_accuracy'] for r in regions]

    x   = np.arange(len(regions))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(max(6, 3 * len(regions)), 5))
    ax.bar(x - w/2, mious, w, label='mIoU',           color='#4361ee')
    ax.bar(x + w/2, paccs, w, label='Pixel Accuracy', color='#4cc9f0')
    for i, (m, p) in enumerate(zip(mious, paccs)):
        ax.text(i - w/2, m + 0.01, f'{m:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + w/2, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(regions, rotation=20, ha='right')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('mIoU and Pixel Accuracy per Region')
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    checkpoint_path: str,
    in_channels: int,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    model = GEESegFormer(
        in_channels       = in_channels,
        num_classes       = num_classes,
        segformer_weights = SEGFORMER_WEIGHTS,
    )
    ckpt  = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  Warning — missing keys: {missing}')
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate GEESegFormer — CL-aware.')
    parser.add_argument('--checkpoint',   required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--chips-dir',    required=True, help='Root chip directory')
    parser.add_argument('--regions',      nargs='+', required=True, help='Region(s) to evaluate on')
    parser.add_argument('--output-dir',   required=True, help='Directory to write results')
    parser.add_argument('--baseline',     default=None,
                        help='Optional baseline checkpoint for forgetting analysis')
    parser.add_argument('--split',        default='val', choices=['train', 'val'],
                        help='Dataset split to evaluate (default: val)')
    parser.add_argument('--mode',         default='full',
                        choices=['rgb', 'rgbn', 'embeddings', 'full'])
    parser.add_argument('--batch-size',   type=int, default=4)
    parser.add_argument('--num-workers',  type=int, default=4)
    parser.add_argument('--n-samples',    type=int, default=6,
                        help='Number of sample prediction images to save per region')
    parser.add_argument('--seed',         type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    with open(TAXONOMY_PATH) as f:
        tax = yaml.safe_load(f)
    num_classes  = tax['num_classes']
    ignore_index = tax['ignore_index']
    class_names  = tax['class_names']

    in_channels = len(BAND_GROUPS[args.mode])
    has_rgb     = args.mode in ('rgb', 'rgbn', 'full')

    print(f'\nLoading checkpoint: {args.checkpoint}')
    model = load_model(args.checkpoint, in_channels, num_classes, device)

    baseline_model   = None
    baseline_metrics = {}
    if args.baseline:
        print(f'Loading baseline: {args.baseline}')
        baseline_model = load_model(args.baseline, in_channels, num_classes, device)

    chips_root   = Path(args.chips_dir)
    all_metrics  = {}

    for region in args.regions:
        print(f'\n── Region: {region} ──')
        region_dir = chips_root / region / args.split
        if not region_dir.exists():
            print(f'  WARNING: {region_dir} not found — skipping')
            continue

        ds = CachedChipDataset(str(region_dir), augment=False)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=(device.type == 'cuda'),
        )
        print(f'  Chips: {len(ds):,}  |  Batches: {len(loader)}')

        cm      = build_confusion_matrix(model, loader, num_classes, ignore_index, device)
        metrics = metrics_from_cm(cm, class_names)
        all_metrics[region] = metrics

        print(f'  mIoU: {metrics["miou"]:.4f}  |  PixAcc: {metrics["pixel_accuracy"]:.4f}')
        for name in class_names:
            iou = metrics['class_iou'][name]
            print(f'    {name:15}: IoU={iou:.3f}' if not np.isnan(iou) else f'    {name:15}: IoU=N/A')

        with open(out_dir / f'metrics_{region}.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        plot_confusion_matrix(
            cm, class_names,
            out_dir / f'confusion_matrix_{region}.png',
        )
        plot_class_metrics(
            metrics, class_names, region,
            out_dir / f'class_metrics_{region}.png',
        )
        print(f'  Saving {args.n_samples} sample predictions...')
        plot_samples(
            ds, model, num_classes, ignore_index, class_names, device,
            out_dir / f'samples_{region}.png',
            n_samples=args.n_samples,
            has_rgb=has_rgb,
        )

        if baseline_model is not None:
            print(f'  Running baseline on {region}...')
            cm_base = build_confusion_matrix(baseline_model, loader, num_classes, ignore_index, device)
            baseline_metrics[region] = metrics_from_cm(cm_base, class_names)

    # ── Multi-region plots ────────────────────────────────────────────────────
    if len(all_metrics) > 1:
        plot_cl_comparison(all_metrics, class_names, out_dir / 'cl_comparison.png')
        print('\nSaved cl_comparison.png')

    plot_miou_summary(all_metrics, out_dir / 'miou_summary.png')

    if baseline_metrics:
        plot_forgetting(all_metrics, baseline_metrics, class_names, out_dir / 'forgetting.png')
        print('Saved forgetting.png')

    # ── Combined JSON ─────────────────────────────────────────────────────────
    summary = {
        'checkpoint':  args.checkpoint,
        'baseline':    args.baseline,
        'regions':     args.regions,
        'split':       args.split,
        'per_region':  all_metrics,
    }
    if baseline_metrics:
        forgetting_bwt = {}
        for region in baseline_metrics:
            if region in all_metrics:
                delta = all_metrics[region]['miou'] - baseline_metrics[region]['miou']
                forgetting_bwt[region] = round(delta, 4)
        summary['backward_transfer'] = forgetting_bwt
        print(f'\nBackward Transfer (mIoU delta): {forgetting_bwt}')

    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nAll results saved to: {out_dir.resolve()}')


if __name__ == '__main__':
    main()
