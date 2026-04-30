#!/usr/bin/env bash
# GCP VM setup — run once after creating the instance.
# Tested on: Deep Learning VM (Debian, CUDA 12.x pre-installed).
# For a plain Debian VM, uncomment the CUDA section below.
#
# Prerequisites: run these manually BEFORE this script:
#   sudo apt-get update && sudo apt-get install -y git
#   git clone https://github.com/UCSD-E4E/ml-mangrove.git
#   cd ml-mangrove
set -euo pipefail

CONDA_ENV="mangrove"
PYTHON_VER="3.11"
REPO_SSH="git@github.com:UCSD-E4E/ml-mangrove.git"
REPO_DIR="$HOME/ml-mangrove"

echo "=== Step 1: System packages ==="
sudo apt-get update -q
sudo apt-get install -y --no-install-recommends \
    git wget curl \
    libgdal-dev gdal-bin \
    libgl1

# Uncomment if NOT using a Deep Learning VM image (plain Debian):
# echo "=== Step 1b: CUDA 12.1 ==="
# wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt-get update -q
# sudo apt-get install -y cuda-toolkit-12-1

echo ""
echo "=== Step 2: Miniconda ==="
if ! command -v conda &>/dev/null; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda init bash
    source ~/.bashrc
else
    eval "$(conda shell.bash hook)"
    echo "  conda already installed: $(conda --version)"
fi

echo ""
echo "=== Step 3: Conda environment ($CONDA_ENV) ==="
if conda env list | grep -q "^$CONDA_ENV "; then
    echo "  Environment already exists — skipping create."
else
    conda create -y -n "$CONDA_ENV" python="$PYTHON_VER"
fi
conda activate "$CONDA_ENV"

echo ""
echo "=== Step 4: Python packages ==="
# PyTorch — matches CUDA 12.1 on most GCP Deep Learning VMs
# Adjust cu121 → cu118 if your driver uses CUDA 11.8
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install \
    rasterio \
    "transformers>=4.40" \
    pyyaml \
    numpy \
    scipy \
    tqdm \
    matplotlib \
    google-cloud-storage \
    timm \
    einops

echo ""
echo "=== Step 5: Clone repo ==="
if [ ! -d "$REPO_DIR" ]; then
    git clone "$REPO_SSH" "$REPO_DIR"
else
    echo "  Repo already cloned at $REPO_DIR — pulling latest."
    git -C "$REPO_DIR" pull --ff-only
fi

echo ""
echo "=== Done ==="
echo "Activate env : conda activate $CONDA_ENV"
echo "Repo         : $REPO_DIR"
echo "Verify GPU   : python -c \"import torch; print(torch.cuda.get_device_name(0))\""
