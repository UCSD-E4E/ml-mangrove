"""
GEE SegFormer training script — single region or continual learning with replay.
Run on the GCP VM after prepare_chips.py has completed.

Examples:
  # Base training (Florida)
  python train.py \\
      --chips-dir /data/chips \\
      --region florida \\
      --experiment gee_florida_v1 \\
      --epochs 100

  # Continual learning (Brazil, replaying Florida)
  python train.py \\
      --chips-dir /data/chips \\
      --region brazil \\
      --experiment gee_brazil_cl_v1 \\
      --resume /data/experiments/gee_florida_v1/best_model.pth \\
      --replay-regions florida \\
      --replay-fraction 0.3 \\
      --epochs 100

  # Resume an interrupted run
  python train.py \\
      --chips-dir /data/chips \\
      --region brazil \\
      --experiment gee_brazil_cl_v1 \\
      --resume /data/experiments/gee_brazil_cl_v1/latest_checkpoint.pth \\
      --replay-regions florida \\
      --replay-fraction 0.3 \\
      --epochs 100
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset

_GEE_ROOT = Path(__file__).parent
sys.path.insert(0, str(_GEE_ROOT))
sys.path.insert(0, str(_GEE_ROOT.parent.parent / 'DroneClassification'))

from gee_dataset import BAND_GROUPS, CachedChipDataset, GEESegFormer   # noqa: E402
from models.loss import JaccardLoss                                      # noqa: E402
from training_utils import TrainingSession                               # noqa: E402

TAXONOMY_PATH    = _GEE_ROOT / 'config' / 'taxonomy.yaml'
EXPERIMENTS_ROOT = _GEE_ROOT / 'experiments'
SEGFORMER_WEIGHTS = 'nvidia/segformer-b4-finetuned-ade-512-512'


# ─────────────────────────────────────────────────────────────────────────────
# Replay-mixed dataset
# ─────────────────────────────────────────────────────────────────────────────

class ReplayMixDataset(Dataset):
    """
    Mixes current-region chips with replay chips from prior regions.
    Each __getitem__ samples from replay with probability `replay_fraction`,
    otherwise from the current region — no custom sampler needed.
    """

    def __init__(
        self,
        current: CachedChipDataset,
        replay_datasets: list[CachedChipDataset],
        replay_fraction: float,
    ):
        self.current          = current
        self.replay           = ConcatDataset(replay_datasets) if replay_datasets else None
        self.replay_fraction  = replay_fraction if replay_datasets else 0.0

    def __len__(self) -> int:
        return len(self.current)

    def __getitem__(self, idx: int):
        if self.replay is not None and random.random() < self.replay_fraction:
            return self.replay[random.randint(0, len(self.replay) - 1)]
        return self.current[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Class weights
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(
    dataset: CachedChipDataset,
    num_classes: int,
    ignore_index: int,
    sample_every: int = 1,
) -> torch.Tensor:
    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    image_counts = np.zeros(num_classes, dtype=np.int64)

    for i in range(0, len(dataset), sample_every):
        _, label = dataset[i]
        lbl = label.numpy()
        total_valid = int((lbl != ignore_index).sum())
        for c in range(num_classes):
            n = int((lbl == c).sum())
            pixel_counts[c] += n
            if n > 0:
                image_counts[c] += total_valid

    freqs = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        if image_counts[c] > 0:
            freqs[c] = pixel_counts[c] / image_counts[c]

    present     = freqs > 0
    median_freq = np.median(freqs[present]) if present.any() else 1.0
    weights     = np.ones(num_classes, dtype=np.float64)
    weights[present] = median_freq / freqs[present]
    return torch.FloatTensor(weights)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Train GEESegFormer on chipped tiles.')
    parser.add_argument('--chips-dir',       required=True, help='Root dir of .npz chips')
    parser.add_argument('--region',          required=True, help='Region to train on')
    parser.add_argument('--experiment',      required=True, help='Experiment name')
    parser.add_argument('--epochs',          type=int, default=100)
    parser.add_argument('--batch-size',      type=int, default=4)
    parser.add_argument('--lr',              type=float, default=5e-5)
    parser.add_argument('--weight-decay',    type=float, default=0.01)
    parser.add_argument('--mode',            default='full',
                        choices=['rgb', 'rgbn', 'embeddings', 'full'])
    parser.add_argument('--patch-size',      type=int, default=512)
    parser.add_argument('--num-workers',     type=int, default=4)
    parser.add_argument('--resume',          default=None,
                        help='Path to checkpoint (.pth) to resume or warm-start from')
    parser.add_argument('--replay-regions',  nargs='+', default=[],
                        help='Region names to use as replay buffer (continual learning)')
    parser.add_argument('--replay-dir',      default='/data/replay',
                        help='Root dir of replay buffers (default: /data/replay)')
    parser.add_argument('--replay-fraction', type=float, default=0.1,
                        help='Fraction of each batch sampled from replay buffer')
    parser.add_argument('--bucket',          default=None,
                        help='GCS bucket to upload best checkpoint after training')
    parser.add_argument('--seed',            type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    if device.type == 'cuda':
        print(f'GPU    : {torch.cuda.get_device_name(0)}')

    # ── Taxonomy ──────────────────────────────────────────────────────────────
    with open(TAXONOMY_PATH) as f:
        tax = yaml.safe_load(f)
    num_classes  = tax['num_classes']
    ignore_index = tax['ignore_index']
    class_names  = tax['class_names']
    print(f'Classes: {num_classes}  {class_names}')

    # ── Datasets ──────────────────────────────────────────────────────────────
    chips_root = Path(args.chips_dir)
    region_dir = chips_root / args.region

    train_ds = CachedChipDataset(str(region_dir / 'train'), augment=True)
    val_ds   = CachedChipDataset(str(region_dir / 'val'),   augment=False)
    print(f'Train chips: {len(train_ds):,}  |  Val chips: {len(val_ds):,}')

    # ── Replay buffer ─────────────────────────────────────────────────────────
    replay_datasets = []
    if args.replay_regions:
        replay_root = Path(args.replay_dir)
        for rname in args.replay_regions:
            rdir = replay_root / rname / 'train'
            if not rdir.exists():
                print(f'WARNING: replay dir not found, skipping: {rdir}')
                continue
            rd = CachedChipDataset(str(rdir), augment=True)
            replay_datasets.append(rd)
            print(f'Replay [{rname}]: {len(rd):,} chips')

    mixed_train = ReplayMixDataset(train_ds, replay_datasets, args.replay_fraction)
    if replay_datasets:
        print(f'Replay fraction: {args.replay_fraction:.0%} per batch')

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        mixed_train,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = (device.type == 'cuda'),
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = (device.type == 'cuda'),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    in_channels = len(BAND_GROUPS[args.mode])
    model = GEESegFormer(
        in_channels       = in_channels,
        num_classes       = num_classes,
        segformer_weights = SEGFORMER_WEIGHTS,
        patch_size        = args.patch_size,
    )

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        # Support both full checkpoint dicts and bare state dicts
        state = ckpt.get('model_state_dict', ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f'Missing keys on resume (new classes?): {missing}')
        print(f'Resumed from: {args.resume}')

    # ── Class weights ─────────────────────────────────────────────────────────
    weights_path = region_dir / 'class_weights.npy'
    if weights_path.exists():
        class_weights = torch.FloatTensor(np.load(weights_path))
        print('Loaded cached class weights.')
    else:
        print('Computing class weights...')
        class_weights = compute_class_weights(train_ds, num_classes, ignore_index)
        np.save(weights_path, class_weights.numpy())

    for i, (name, w) in enumerate(zip(class_names, class_weights)):
        print(f'  {name:15}: weight={w:.3f}')

    # ── Loss / optimizer / scheduler ─────────────────────────────────────────
    loss_fn = JaccardLoss(
        num_classes  = num_classes,
        weight       = class_weights.to(device),
        alpha        = 0.4,
        ignore_index = ignore_index,
        smooth       = 1e-6,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = args.epochs,
        eta_min = args.lr * 0.01,
    )

    # ── TrainingSession ───────────────────────────────────────────────────────
    exp_dir = EXPERIMENTS_ROOT / args.experiment
    trainer = TrainingSession(
        model           = model,
        trainLoader     = train_loader,
        testLoader      = val_loader,
        lossFunc        = loss_fn,
        init_lr         = args.lr,
        num_epochs      = args.epochs,
        device          = device,
        class_names     = class_names,
        optimizer       = optimizer,
        scheduler       = scheduler,
        experiment_name = args.experiment,
        save_checkpoints= True,
        ignore_index    = ignore_index,
        metric_mode     = 'segmentation',
    )

    print(f'\nExperiment dir : {trainer.experiment_dir.resolve()}')
    print(f'Epochs         : {args.epochs}')
    print(f'Batch size     : {args.batch_size}')
    print(f'Train batches  : {len(train_loader)}')
    print(f'Val batches    : {len(val_loader)}\n')

    trainer.learn()

    # ── Upload best checkpoint to GCS ─────────────────────────────────────────
    if args.bucket:
        best_ckpt = trainer.experiment_dir / 'best_model.pth'
        if best_ckpt.exists():
            try:
                from google.cloud import storage
                client    = storage.Client()
                bkt       = client.bucket(args.bucket)
                gcs_path  = f'checkpoints/{args.experiment}/best_model.pth'
                blob      = bkt.blob(gcs_path)
                blob.upload_from_filename(str(best_ckpt))
                print(f'Uploaded best checkpoint → gs://{args.bucket}/{gcs_path}')
            except Exception as e:
                print(f'WARNING: GCS upload failed: {e}')


if __name__ == '__main__':
    main()
