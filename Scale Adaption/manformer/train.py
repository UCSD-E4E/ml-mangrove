"""
train.py — Manformer entry point.

Wires together:
  - L2HNet CNN backbone  (from ../Paraformer/networks/)
  - ViT-B/16 Transformer (from ../Paraformer/networks/)
  - Mangrove trainer     (trainer.py)
  - 6-class taxonomy     (utils.py)

Usage:
    python train.py \\
        --train_list dataset/florida_manformer_train.csv \\
        --val_list   dataset/florida_manformer_val.csv \\
        --savepath   experiments/run_001 \\
        --gpu        0

    # Larger model (normal mode, 2× CNN width):
    python train.py ... --CNN_width 128

    # Train without ViT pretrained weights (not recommended, slower convergence):
    python train.py ... --pretrained_vit ""
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn

# ── Reuse Paraformer network code directly (no copy needed) ──────────────────
_PARAFORMER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'Paraformer')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # local modules first
sys.path.insert(1, _PARAFORMER_DIR)

from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling_L2HNet import L2HNet

import utils
from trainer import trainer_manformer

# ── Argument parser ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Manformer: NAIP (1 m) + ESA label (10 m) → HR mangrove segmentation'
)
parser.add_argument('--train_list',
                    type=str, required=True,
                    help='Path to train CSV with columns image_fn, label_fn')
parser.add_argument('--val_list',
                    type=str, default='',
                    help='Path to val CSV (optional; enables eval + visualization)')
parser.add_argument('--savepath',
                    type=str, required=True,
                    help='Directory for checkpoints, logs, and prediction images')
parser.add_argument('--gpu',
                    type=str, default='0',
                    help='CUDA device index (e.g. "0" or "0,1")')
parser.add_argument('--max_epochs',
                    type=int, default=100)
parser.add_argument('--batch_size',
                    type=int, default=10,
                    help='Chips per batch; reduce if GPU OOM')
parser.add_argument('--base_lr',
                    type=float, default=0.01)
parser.add_argument('--seed',
                    type=int, default=1234)
parser.add_argument('--CNN_width',
                    type=int, default=64,
                    help='L2HNet feature width: 64 = light mode, 128 = normal mode')
parser.add_argument('--save_interval',
                    type=int, default=20,
                    help='Save a checkpoint every N epochs')
parser.add_argument('--eval_interval',
                    type=int, default=20,
                    help='Run evaluation + visualization every N epochs')
parser.add_argument('--pretrained_vit',
                    type=str,
                    default=os.path.join(_PARAFORMER_DIR, 'networks',
                                         'pre-train_model', 'imagenet21k',
                                         'ViT-B_16.npz'),
                    help='Path to ImageNet-21k ViT-B/16 .npz weights. '
                         'Download link in Scale Adaption/Paraformer/README.md. '
                         'Pass an empty string to train from scratch.')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if __name__ == '__main__':
    # ── Reproducibility ───────────────────────────────────────────────────────
    cudnn.benchmark     = True
    cudnn.deterministic = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.makedirs(args.savepath, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    VIT_PATCH_SIZE = 16
    IMG_SIZE       = 224

    config_vit = CONFIGS_ViT_seg['ViT-B_16']
    config_vit.n_classes    = utils.NUM_CLASSES                     # 6
    config_vit.patches.grid = (IMG_SIZE // VIT_PATCH_SIZE,
                                IMG_SIZE // VIT_PATCH_SIZE)          # (14, 14)

    # image_band=4: NAIP RGBN (vs. the default 3-band RGB in original Paraformer)
    backbone = L2HNet(width=args.CNN_width, image_band=4)
    net = ViT_seg(
        config_vit,
        backbone=backbone,
        img_size=IMG_SIZE,
        num_classes=utils.NUM_CLASSES,
    ).cuda()

    # ── Pretrained ViT weights ────────────────────────────────────────────────
    if args.pretrained_vit and os.path.exists(args.pretrained_vit):
        print(f'Loading ViT-B/16 ImageNet-21k weights from:\n  {args.pretrained_vit}')
        net.load_from(weights=np.load(args.pretrained_vit))
    else:
        if args.pretrained_vit:
            print(f'WARNING: pretrained weights not found at:\n  {args.pretrained_vit}')
            print('  Download link: see Scale Adaption/Paraformer/README.md')
        print('Training ViT branch from random initialisation (slower convergence).')

    # ── Summary ───────────────────────────────────────────────────────────────
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f'\nModel          : Paraformer  (L2HNet width={args.CNN_width}, ViT-B/16)')
    print(f'Parameters     : {n_params:.1f} M')
    print(f'Input          : 4-band NAIP RGBN, {IMG_SIZE}×{IMG_SIZE} chips at 1 m')
    print(f'Classes ({utils.NUM_CLASSES})    : {utils.CLASS_NAMES}')
    print(f'Train CSV      : {args.train_list}')
    print(f'Val CSV        : {args.val_list or "(none — eval disabled)"}')
    print(f'Save path      : {args.savepath}')
    print(f'Max epochs     : {args.max_epochs}')
    print(f'Batch size     : {args.batch_size}')
    print(f'Base LR        : {args.base_lr}')
    print()

    trainer_manformer(args, net, args.savepath)
