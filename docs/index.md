# ML-Mangrove Documentation

Machine learning platform for mangrove ecosystem monitoring from aerial and satellite imagery.

**Developed by**: UCSD Engineers for Exploration (E4E)

---

## Quick Navigation

### Getting Started
- [Environment Setup](setup/environment.md) - Python environment, dependencies, WSL2/CUDA setup
- [ArcGIS Installation](setup/arcgis.md) - Setting up the ArcGIS Pro toolbox

### User Guides
- [Data Preparation](guides/data_prep.md) - Processing geospatial imagery for training
- [Model Training](guides/training.md) - Training segmentation models
- [Inference](guides/inference.md) - Running predictions on new imagery

### Architecture
- [Model Architectures](architecture/models.md) - ResNet-UNet, SegFormer, DeepLab, MambaUNet
- [Loss Functions](architecture/losses.md) - Jaccard, Boundary IoU, Weighted CE
- [Mamba Integration](../DroneClassification/testing/mamba/mamba.md) - State-space model experiments

### Datasets
- [LandCover.ai](datasets/landcover_ai.md) - Aerial land cover segmentation dataset

---

## Project Overview

### Core Capabilities

| Capability | Description | Status |
|------------|-------------|--------|
| **Aerial Segmentation** | Pixel-level mangrove detection from drone imagery | Active |
| **Infrastructure Detection** | Identify roads/buildings threatening mangroves | Active |
| **ArcGIS Toolbox** | User-friendly tools for environmental scientists | Active |
| **Mamba Integration** | State-space models for efficient inference | In Progress |

### Directory Structure

```
ml-mangrove/
├── DroneClassification/     # Main ML pipeline
│   ├── data/                # Data processing utilities
│   ├── models/              # Model architectures & losses
│   ├── training_utils/      # Training framework
│   └── testing/             # Experimental work
├── ARC_Package/             # ArcGIS Pro toolbox
├── docs/                    # Documentation (you are here)
└── archive/                 # Historical approaches
```

### Current Models

| Model | Best For | Parameters | Typical IoU |
|-------|----------|------------|-------------|
| ResNet18-UNet | Mangrove detection | 15.9M | 82-85% |
| SegFormer B0 | Fast inference | ~3.7M | Good |
| SegFormer B2 | Infrastructure detection | ~27M | Better |
| DeepLabv3+ | Multi-class segmentation | ~40M | 81-84% |
| MambaUNet | Efficient alternative | 622K | Experimental |

---

## Contributing

See the main [README](../README.md) for contribution guidelines.
