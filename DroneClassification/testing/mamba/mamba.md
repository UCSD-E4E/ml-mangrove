# Mamba Integration for Mangrove Segmentation

**Goal**: Integrate Mamba state-space models into the mangrove segmentation pipeline to improve performance over existing approaches, and eventually scale to satellite imagery.

---

## Quick Start

### Step 1: Set Up WSL2 Environment

Follow the complete setup guide to install and configure all dependencies:

**Follow: [WSL2 Setup Guide](docs/WSL2_SETUP_GUIDE.md)**

This will install:
- WSL2 with Ubuntu
- CUDA Toolkit 12.6
- Python 3.10 environment (`mamba-env`)
- PyTorch 2.4.1 with CUDA support
- mamba-ssm (built from source)

---

### Step 2: Verify Installation

After completing the setup guide, test that everything works:

**Run: [01_test_mamba.ipynb](01_test_mamba.ipynb)**

This notebook verifies:
- ✓ mamba-ssm imports successfully
- ✓ CUDA is accessible from PyTorch
- ✓ Mamba forward pass works on GPU
- ✓ Gradient computation works (training readiness)

**How to run**:
1. Open VSCode and connect to WSL (green button bottom-left)
2. Open `01_test_mamba.ipynb`
3. Select kernel: **Python (Mamba-WSL2)** or **Python 3.10.19 ('mamba-env')**
4. Run all cells

---

### Step 3: Build Mamba-UNet Architecture

**Run: [02_build_mamba_unet.ipynb](02_build_mamba_unet.ipynb)**

This notebook builds and tests the complete MambaUNet architecture:
- **PatchEmbedding**: Converts 512×512 images to 1024 patch tokens (Conv2d with stride=16)
- **MambaEncoder**: Processes tokens with 3 Mamba layers (pre-norm + residual)
- **Decoder**: Upsamples back to 512×512 segmentation mask (4 ConvTranspose2d layers)
- **End-to-end pipeline**: image → patches → Mamba processing → mask

**Tested**: ✓ Works end-to-end with no errors

---

### Step 4: Train on Landcover.ai Dataset

**Run: [03_train_mamba_unet.ipynb](03_train_mamba_unet.ipynb)**

This notebook trains MambaUNet on real aerial imagery and compares with ResNet_UNet:
- Load Landcover.ai v1 dataset (41 GeoTIFFs, 512×512 tiles, 5 land cover classes)
- Train MambaUNet with weighted CrossEntropyLoss (handles class imbalance)
- Train ResNet_UNet baseline with same data for direct comparison
- Compare training curves, loss values, parameter counts, and validation IoU
- Automatic checkpointing and metrics logging for both models

**Status**: ✓ Complete - 2 epochs trained successfully
- **MambaUNet**: Val IoU = 0.1127 (622K params), Epoch 2 best model
- **ResNet_UNet**: Val IoU = 0.2448 (15.9M params), Epoch 2 best model - 2.17× higher IoU
- ResNet achieves better accuracy but MambaUNet is 25.5× smaller with faster inference

---

## Next Steps

We have couple options to choose from:

**Option 1: Improve MambaUNet Performance**
- Extend training to 10-20 epochs to reach higher IoU
- Experiment with different architectures (more Mamba layers, larger embeddings)
- Fine-tune hyperparameters (learning rate, batch size, weight decay)
- Apply data augmentation (rotations, flips, color jittering) to improve generalization

**Option 2: Extended Baseline Comparison**
- Train both models for more epochs (20-50) to convergence
- Compare inference speed and memory usage on test set
- Evaluate on mangrove-specific imagery to test domain transfer
- Generate performance curves and statistical comparisons

**Option 3: Human Infrastructure Detection**
- Pivot to human-in-the-loop infrastructure detection task
- Fine-tune existing models (MambaUNet or ResNet) on new labeled data
- Integrate into inference pipeline for real-world deployment
- Focus on practical accuracy and computational efficiency for edge devices

**Dataset**: Landcover.ai v1
- **Type**: Aerial land cover segmentation (similar domain to mangrove detection)
- **Classes**: Background (0), Building (1), Woodland (2), Water (3), Road (4)
- **Location**: `DroneClassification/testing/mamba/landcover.ai.v1/`
- **Info**: [LANDCOVER_AI_README.md](LANDCOVER_AI_README.md)
- **Added to .gitignore**: Dataset not tracked in git (1.5GB images + 39MB masks). Follow [LANDCOVER_AI_README.md](LANDCOVER_AI_README.md) to download the dataset.

---

## Architecture Overview

### Model: MambaUNet

```
Input: (B, 3, 512, 512)
  ↓
PatchEmbedding (Conv2d 16×16 stride)
  ↓ (B, 1024, 128)
MambaEncoder (3 Mamba layers)
  ↓ (B, 1024, 128)
Decoder (4× ConvTranspose2d upsampling)
  ↓
Output: (B, 1, 512, 512)
```

### Key Components

1. **PatchEmbedding** (98KB params)
   - Conv2d(3 → 128, kernel=16, stride=16)
   - Converts image to patch tokens

2. **MambaEncoder** (varies)
   - 3 layers of: LayerNorm + Mamba + Residual
   - Processes tokens sequentially
   - Pre-norm architecture (stable training)

3. **Decoder** (varies)
   - 4 stages of ConvTranspose2d upsampling
   - Progressive channel reduction: 128 → 64 → 32 → 16 → 8 → 1
   - Recovers 512×512 resolution
   - Configured for 5 classes (Landcover.ai: Building, Woodland, Water, Road)
   
---

## Files in This Directory

| File | Purpose | Status |
|------|---------|--------|
| `mamba.md` | This file - integration overview | ✓ Updated |
| `01_test_mamba.ipynb` | Verify mamba-ssm installation | ✓ Complete |
| `02_build_mamba_unet.ipynb` | Build and test architecture | ✓ Complete |
| `03_train_mamba_unet.ipynb` | Train MambaUNet and ResNet_UNet | ✓ Complete |
| `experiments/MambaUNet_Landcover_5class/` | MambaUNet training results | ✓ Logged |
| `experiments/ResNet_UNet_Landcover_5class/` | ResNet_UNet training results | ✓ Logged |
| `LANDCOVER_AI_README.md` | Dataset information & structure | ✓ Updated |
| `docs/WSL2_SETUP_GUIDE.md` | Complete environment setup | ✓ Reference |
| `landcover.ai.v1/` | Landcover.ai v1 dataset | 📊 Not tracked |
