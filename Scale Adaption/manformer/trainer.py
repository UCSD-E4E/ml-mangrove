"""
trainer.py — Manformer training loop, evaluation, and visualization.

Adapts Paraformer's trainer (Li et al., CVPR 2024) for the 6-class mangrove
taxonomy and NAIP + ESA WorldCover data.

Key differences from the original Paraformer trainer:
  - IGNORE_INDEX = 255 (not 0) — unknown ESA values are excluded from loss
  - Masked CE uses 255 as the ignored "disagree" value (not 0)
  - evaluate()           — per-class IoU, mIoU, pixel accuracy on a val set
  - visualize_predictions() — 3-panel NAIP | ESA label | prediction figure
"""

import logging
import os
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm

import utils

# Minimum fraction of pixels in a chip that must carry a known ESA label.

# Must be module-level (not nested) so Windows multiprocessing can pickle it
def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════════════════

class StreamingGeospatialDataset(IterableDataset):
    """
    Streams random chips from large GeoTIFF pairs (HR image + LR label).

    Identical in design to Paraformer's StreamingGeospatialDataset; kept here
    so the manformer directory is self-contained.
    """

    def __init__(self, imagery_fns, label_fns, chip_size=224,
                 num_chips_per_tile=50, windowed_sampling=True,
                 image_transform=None, label_transform=None,
                 nodata_check=None, verbose=False):
        self.fns          = list(zip(imagery_fns, label_fns))
        self.chip_size    = chip_size
        self.num_chips_per_tile = num_chips_per_tile
        self.windowed_sampling  = windowed_sampling
        self.image_transform    = image_transform
        self.label_transform    = label_transform
        self.nodata_check       = nodata_check
        self.verbose            = verbose

    def _stream_tile_fns(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers

        fns = list(self.fns)
        if worker_id == 0:
            np.random.shuffle(fns)

        N = len(fns)
        per_worker = int(np.ceil(N / num_workers))
        lo, hi = worker_id * per_worker, min(N, (worker_id + 1) * per_worker)
        for idx in range(lo, hi):
            yield fns[idx]

    def _stream_chips(self):
        for img_fn, label_fn in self._stream_tile_fns():
            num_skipped = 0

            # Load entire tile into memory upfront.
            # GEE GeoTIFFs are strip-based (not tiled), so per-chip windowed reads
            # are extremely slow — they decompress large strips for every small window.
            # For 40-70 MB NAIP tiles this full-load approach is much faster.
            try:
                with rasterio.open(img_fn) as f:
                    img_data = np.rollaxis(f.read(), 0, 3)      # (H, W, C) uint8
                with rasterio.open(label_fn) as f:
                    lbl_data = f.read(1)                         # (H, W) uint8
            except Exception as e:
                print(f'WARNING: Could not read {img_fn}: {e}')
                continue

            # Robustly handle 1-pixel mismatches from GEE reprojection
            H = min(img_data.shape[0], lbl_data.shape[0])
            W = min(img_data.shape[1], lbl_data.shape[1])
            img_data = img_data[:H, :W]
            lbl_data = lbl_data[:H, :W]

            if self.verbose:
                print(f'Loaded {os.path.basename(img_fn)}: {W}×{H}, {img_data.shape[2]} bands')

            if H <= self.chip_size or W <= self.chip_size:
                print(f'WARNING: tile too small ({W}×{H}) for chip size {self.chip_size}, skipping.')
                continue

            for _ in range(self.num_chips_per_tile):
                x = np.random.randint(0, W - self.chip_size)
                y = np.random.randint(0, H - self.chip_size)

                img    = img_data[y:y + self.chip_size, x:x + self.chip_size]   # (H,W,C)
                labels = lbl_data[y:y + self.chip_size, x:x + self.chip_size]   # (H,W)

                if self.nodata_check is not None:
                    if self.nodata_check(img, labels):
                        num_skipped += 1
                        continue

                if self.image_transform is not None:
                    img = self.image_transform(img)
                else:
                    img = torch.from_numpy(np.rollaxis(img, 2, 0).astype(np.float32))

                if self.label_transform is not None:
                    labels = self.label_transform(labels)
                else:
                    labels = torch.from_numpy(labels.astype(np.int64))

                yield img, labels

            if num_skipped > 0 and self.verbose:
                print(f'  Skipped {num_skipped}/{self.num_chips_per_tile} chips (nodata filter)')

    def __iter__(self):
        return iter(self._stream_chips())


# ── Transforms ────────────────────────────────────────────────────────────────

def image_transforms(img: np.ndarray) -> torch.Tensor:
    """
    (H, W, 4) uint8 NAIP → (4, H, W) float32 z-score normalized tensor.
    """
    img = (img.astype(np.float32) - utils.IMAGE_MEANS) / (utils.IMAGE_STDS + 1e-8)
    img = np.rollaxis(img, 2, 0)          # (4, H, W)
    return torch.from_numpy(img.copy())


def label_transforms(labels: np.ndarray) -> torch.Tensor:
    """
    (H, W) raw ESA uint8 values → (H, W) int64 class indices.
    Unknown ESA values become IGNORE_INDEX (255).
    """
    remapped = utils.LABEL_CLASS_TO_IDX_MAP[labels.astype(np.uint8)]
    return torch.from_numpy(remapped).long()




# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, val_loader, device='cuda'):
    """
    Evaluate model on val_loader using fused branch output (average of both softmaxes).

    Returns:
        dict with keys: mIoU (float), pixel_acc (float), class_ious (list of float).
        NaN indicates a class was absent from the validation set.
    """
    model.eval()
    confusion = np.zeros((utils.NUM_CLASSES, utils.NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc='  Evaluating', leave=False):
            imgs   = imgs.to(device)
            labels = labels.to(device)

            logits1, logits2 = model(imgs)
            preds = (
                (F.softmax(logits1, dim=1) + F.softmax(logits2, dim=1)) / 2
            ).argmax(dim=1)

            mask       = labels != utils.IGNORE_INDEX
            pred_flat  = preds[mask].cpu().numpy()
            label_flat = labels[mask].cpu().numpy()

            np.add.at(confusion, (label_flat, pred_flat), 1)

    # Per-class IoU
    class_ious = []
    for c in range(utils.NUM_CLASSES):
        tp    = confusion[c, c]
        fp    = confusion[:, c].sum() - tp
        fn    = confusion[c, :].sum() - tp
        denom = tp + fp + fn
        class_ious.append(float(tp) / float(denom) if denom > 0 else float('nan'))

    valid_ious = [x for x in class_ious if not np.isnan(x)]
    miou       = float(np.mean(valid_ious)) if valid_ious else 0.0
    pixel_acc  = float(np.diag(confusion).sum()) / float(confusion.sum() + 1e-8)

    model.train()
    return {'mIoU': miou, 'pixel_acc': pixel_acc, 'class_ious': class_ious}


# ══════════════════════════════════════════════════════════════════════════════
#  Visualization
# ══════════════════════════════════════════════════════════════════════════════

def visualize_predictions(model, val_loader, save_path, epoch,
                          device='cuda', n_samples=3):
    """
    Save a PNG with n_samples rows, each showing:
      [NAIP RGB (1 m)]  |  [ESA Label — colorized]  |  [Prediction — colorized]

    Saved to: <save_path>/predictions_epoch_XXXX.png
    """
    model.eval()
    samples = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            if len(samples) >= n_samples:
                break
            logits1, logits2 = model(imgs.to(device))
            preds = (
                (F.softmax(logits1, dim=1) + F.softmax(logits2, dim=1)) / 2
            ).argmax(dim=1)

            for j in range(min(imgs.shape[0], n_samples - len(samples))):
                naip_norm = imgs[j].numpy()           # (4, H, W), normalized
                label_np  = labels[j].numpy()         # (H, W), class indices
                pred_np   = preds[j].cpu().numpy()    # (H, W), class indices

                # Denormalize RGB channels (0, 1, 2) for display
                rgb = naip_norm[:3].transpose(1, 2, 0)                    # (H,W,3)
                rgb = rgb * utils.IMAGE_STDS[:3] + utils.IMAGE_MEANS[:3]  # undo z-score
                rgb = np.clip(rgb / 255.0, 0.0, 1.0)

                samples.append((rgb, label_np, pred_np))

    n_rows = len(samples)
    if n_rows == 0:
        logging.warning('visualize_predictions: no samples collected, skipping.')
        model.train()
        return

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    col_titles = ['NAIP RGB (1 m)', 'ESA Label (LR — colorized)', 'Prediction (HR — colorized)']
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=11, fontweight='bold')

    for row, (rgb, label, pred) in enumerate(samples):
        axes[row][0].imshow(rgb)
        axes[row][0].axis('off')
        axes[row][1].imshow(utils.make_label_rgb(label))
        axes[row][1].axis('off')
        axes[row][2].imshow(utils.make_label_rgb(pred))
        axes[row][2].axis('off')

    # Shared legend
    patches = [
        mpatches.Patch(facecolor=utils.CLASS_COLORS[i], label=utils.CLASS_NAMES[i])
        for i in range(utils.NUM_CLASSES)
    ]
    fig.legend(handles=patches, loc='lower center', ncol=utils.NUM_CLASSES,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(f'Epoch {epoch}', fontsize=13, y=1.01)
    plt.tight_layout()

    save_fn = os.path.join(save_path, f'predictions_epoch_{epoch:04d}.png')
    plt.savefig(save_fn, bbox_inches='tight', dpi=100)
    plt.close()
    logging.info(f'Saved visualization → {save_fn}')
    model.train()


# ══════════════════════════════════════════════════════════════════════════════
#  Training loop
# ══════════════════════════════════════════════════════════════════════════════

def trainer_manformer(args, model, snapshot_path):
    """
    Full Paraformer training loop adapted for mangrove + NAIP data.

    Dual-branch masked CE loss:
      loss = 0.5 * CE(CNN_branch, LR_label)
           + 0.5 * CE(ViT_branch, mask_label)

    where mask_label retains LR_label only at pixels where the CNN branch
    agrees with it; all other pixels are set to IGNORE_INDEX (255).
    """
    # ── Logging ───────────────────────────────────────────────────────────────
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'log.txt'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr    = args.base_lr
    batch_size = args.batch_size

    # ── Training dataset ──────────────────────────────────────────────────────
    CHIPS_PER_TILE = 50
    CHIP_SIZE      = 224

    train_df  = pd.read_csv(args.train_list)
    train_img = train_df['image_fn'].values
    train_lbl = train_df['label_fn'].values

    db_train = StreamingGeospatialDataset(
        imagery_fns=train_img, label_fns=train_lbl,
        chip_size=CHIP_SIZE, num_chips_per_tile=CHIPS_PER_TILE,
        windowed_sampling=True,
        image_transform=image_transforms,
        label_transform=label_transforms,
        nodata_check=None,
    )

    train_loader = DataLoader(
        db_train, batch_size=batch_size, num_workers=0,
        pin_memory=True, worker_init_fn=_worker_init_fn
    )

    # ── Validation dataset (optional) ────────────────────────────────────────
    val_loader = None
    val_list   = getattr(args, 'val_list', '')
    if val_list and os.path.exists(val_list):
        val_df  = pd.read_csv(val_list)
        db_val  = StreamingGeospatialDataset(
            imagery_fns=val_df['image_fn'].values,
            label_fns=val_df['label_fn'].values,
            chip_size=CHIP_SIZE, num_chips_per_tile=25,
            windowed_sampling=True,
            image_transform=image_transforms,
            label_transform=label_transforms,
            nodata_check=None,
        )
        val_loader = DataLoader(db_val, batch_size=batch_size,
                                num_workers=0, pin_memory=True)
        logging.info(f'Val tiles: {len(val_df)}')

    # ── Loss, optimiser, scheduler ───────────────────────────────────────────
    model.train()
    ce_loss   = CrossEntropyLoss(ignore_index=utils.IGNORE_INDEX)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
    writer    = SummaryWriter(os.path.join(snapshot_path, 'log'))

    max_epoch     = args.max_epochs
    n_train       = len(train_img) * CHIPS_PER_TILE
    max_iters     = max_epoch * n_train
    save_interval = getattr(args, 'save_interval', 20)
    eval_interval = getattr(args, 'eval_interval', 20)
    iter_num      = 0

    logging.info(f'Train tiles : {len(train_img)}')
    logging.info(f'~{n_train} samples/epoch  |  max_epochs={max_epoch}  |  max_iters={max_iters}')

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(max_epoch):
        loss_ce1_list, loss_ce2_list = [], []

        for imgs, labels in tqdm(
            train_loader,
            desc=f'Epoch {epoch:4d}',
            total=n_train // batch_size,
        ):
            imgs, labels = imgs.cuda(), labels.cuda()

            logits1, logits2 = model(imgs)

            # Masked CE: ViT branch supervises only pixels where CNN agrees with LR label
            cnn_pred    = F.softmax(logits1, dim=1).argmax(dim=1)
            mask_labels = torch.where(
                cnn_pred == labels,
                labels,
                torch.full_like(labels, utils.IGNORE_INDEX)
            )

            loss_ce1 = ce_loss(logits1, labels)
            loss_ce2 = ce_loss(logits2, mask_labels)
            loss     = 0.5 * loss_ce1 + 0.5 * loss_ce2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Polynomial LR decay (same as original Paraformer)
            lr = base_lr * (1.0 - iter_num / max_iters) ** 0.9
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            loss_ce1_list.append(loss_ce1.item())
            loss_ce2_list.append(loss_ce2.item())
            writer.add_scalar('Loss/CE_CNN',  loss_ce1.item(), iter_num)
            writer.add_scalar('Loss/MCE_ViT', loss_ce2.item(), iter_num)
            iter_num += 1

        avg1  = float(np.mean(loss_ce1_list))
        avg2  = float(np.mean(loss_ce2_list))
        total = 0.5 * avg1 + 0.5 * avg2
        logging.info(
            f'Epoch {epoch:4d} | CE_CNN={avg1:.4f}  MCE_ViT={avg2:.4f}  '
            f'total={total:.4f}  lr={lr:.6f}'
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if epoch % save_interval == 0 or epoch == max_epoch - 1:
            ckpt = os.path.join(snapshot_path, f'epoch_{epoch:04d}.pth')
            torch.save(model.state_dict(), ckpt)
            logging.info(f'  Saved checkpoint → {ckpt}')

        # ── Evaluation + visualization ────────────────────────────────────────
        if val_loader is not None and (
            epoch % eval_interval == 0 or epoch == max_epoch - 1
        ):
            metrics = evaluate(model, val_loader)

            logging.info(
                f'  Val  mIoU={metrics["mIoU"]:.4f}  '
                f'PixelAcc={metrics["pixel_acc"]:.4f}'
            )
            for name, iou in zip(utils.CLASS_NAMES, metrics['class_ious']):
                val_str = f'{iou:.4f}' if not np.isnan(iou) else '   N/A'
                logging.info(f'    {name:<15} IoU={val_str}')
                if not np.isnan(iou):
                    writer.add_scalar(f'IoU/{name}', iou, epoch)
            writer.add_scalar('Val/mIoU',     metrics['mIoU'],     epoch)
            writer.add_scalar('Val/PixelAcc', metrics['pixel_acc'], epoch)

            visualize_predictions(model, val_loader, snapshot_path, epoch)

    writer.close()
    logging.info('Training finished.')
    return 'Training Finished!'
