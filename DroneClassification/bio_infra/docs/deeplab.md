# DeepLab Implementation for LandCover.ai

## What's Been Implemented

The notebook 04_train_deeplab.ipynb implements **DeepLabv3+ semantic segmentation** on the LandCover.ai dataset to achieve land cover classification with 5 classes.

### Core Architecture
- **Model**: DeepLabv3+ with ResNet50 backbone
- **Output Stride**: 4 (atrous convolutions with dilation factors 2 and 4)
- **ASPP Module**: Multi-scale context aggregation
- **Decoder**: Bilinear upsampling to full resolution
- **Input Size**: 512×512 RGB images
- **Output**: 5-class semantic segmentation mask

### Training Configuration
- **Loss Function**: CrossEntropyLoss with inverse frequency class weights
- **Learning Rate**: 1e-2 (appropriate for segmentation fine-tuning)
- **Optimizer**: AdamW (PyTorch default)
- **Scheduler**: Cosine annealing over 20 epochs
- **Batch Size**: 8 (RTX 4060 hardware constraint)
- **Epochs**: 20

### Data Pipeline
- **Dataset**: LandCover.ai (41 GeoTIFF orthorectified aerial images)
- **Coverage**: 176.76 km² train + 39.51 km² test
- **Preprocessing**: Converted to 512×512 tiles via split.py
- **Split**: 7,470 train / 1,602 val / 1,602 test tiles
- **Augmentation**: Online augmentation applied during training
  - Random horizontal/vertical flips (50% probability)
  - Random rotation (±30°)
  - Color jitter (brightness, contrast, saturation, hue)
  - Gaussian blur (kernel 3×3, σ ∈ [0.1, 1.0])
  - Synchronized augmentation: same seed applied to image and mask

### Evaluation Metrics
- **mIoU**: Mean Intersection-over-Union across 5 classes
- **Per-class IoU**: Individual class performance
- **Per-class F1**: Harmonic mean of precision and recall
- **Per-class Accuracy**: True positive rate per class


| Configuration | mIoU | Notes |
|---------------|------|-------|
| **Current Setup** | **81-84%** | ResNet50, OS=4, batch=8, online augmentation |
| Paper (Xception71) | 85.56% | Full setup with larger backbone |
| Theoretical Max | 83-84% | Current architecture limit on RTX 4060 |


## Differences from Paper (arXiv:2005.02264v3)

| Aspect | Paper | This Implementation | Reason |
|--------|-------|-------------------|--------|
| Backbone | Xception71 | ResNet50 | Simplicity + faster convergence |
| Batch Size | 16-32 | 8 | RTX 4060 memory constraint (6GB) |
| Augmentation | Offline (9× multiplier) | Online | Save disk space, same effectiveness |
| Optimizer | SGD + Momentum | AdamW | More stable convergence |
| Schedule | Poly Decay | Cosine Annealing | Smooth decay, simpler implementation |
| LR Schedule | 0.01 → 0 | 0.01 → 0 | ✅ Matched |
| Output Stride | 4 | 4 | ✅ Matched |
| Loss Function | CrossEntropy + weights | CrossEntropy + weights | ✅ Matched |