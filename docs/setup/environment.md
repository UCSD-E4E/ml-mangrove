# Environment Setup

This guide covers setting up your development environment for ml-mangrove.

---

## Quick Start (Standard Setup)

For most users working on Windows/Mac/Linux without Mamba models:

```bash
# Clone the repository
git clone https://github.com/UCSD-E4E/ml-mangrove.git
cd ml-mangrove

# Create conda environment
conda create -n mangrove python=3.11 -y
conda activate mangrove

# Install PyTorch (with CUDA if available)
# Follow https://pytorch.org/ to install torch for your OS
# Example for Windows: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install project dependencies
pip install -r requirements.txt
```

---

## WSL2 + CUDA Setup (For Mamba Models)

Mamba state-space models require specific versions and WSL2 on Windows.

**Full guide**: [WSL2 Setup Guide](../../DroneClassification/testing/mamba/docs/WSL2_SETUP_GUIDE.md)

---

## VSCode Integration

### Standard Setup
1. Install Python extension
2. Select interpreter: `mangrove` conda environment
3. Open notebooks with Jupyter extension

### WSL2 Setup (for Mamba)
1. Install "Remote - WSL" extension
2. Click green button (bottom-left) → "Reopen in WSL"
3. Select kernel: "Python (Mamba-WSL2)"