# Mamba Integration for Mangrove Segmentation

**Goal**: Integrate Mamba state-space models into the mangrove segmentation pipeline to improve performance over existing approaches, and eventually do great on satellite images.

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
-  mamba-ssm imports successfully
-  CUDA is accessible from PyTorch
-  Mamba forward pass works on GPU
-  Gradient computation works (training readiness)

**How to run**:
1. Open VSCode and connect to WSL (green button bottom-left)
2. Open `01_test_mamba.ipynb`
3. Select kernel: **Python (Mamba-WSL2)** or **Python 3.10.19 ('mamba-env')**
4. Run all cells

---
