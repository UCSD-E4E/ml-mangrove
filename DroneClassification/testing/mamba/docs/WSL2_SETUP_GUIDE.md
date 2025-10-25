# WSL2 + CUDA Setup Guide for Mamba Integration

## Key Version Requirements

After extensive testing, these specific versions are required for mamba-ssm to work correctly:

| Component | Version | Why This Version? |
|-----------|---------|-------------------|
| **Python** | 3.10.x | Required by mamba-ssm |
| **PyTorch** | 2.4.1+cu121 | Sweet spot: has required APIs, stable C++ ABI |
| **NumPy** | < 2.0 (e.g., 1.26.4) | mamba-ssm compiled extensions need NumPy 1.x |
| **CUDA Toolkit** | 12.6 (WSL) | Latest stable, backward compatible with cu121 |
| **libstdcxx-ng** | 13.2.0+ | For C++17 support (GLIBCXX_3.4.32) |

**Critical**: mamba-ssm MUST be compiled from source against your PyTorch version using `--no-binary` flags.

---

## Prerequisites

✅ **Before you start, verify:**
1. Windows 10 version 2004+ or Windows 11
2. NVIDIA GPU with CUDA support (you already have this)
3. ~20GB free disk space
4. Administrator access to Windows

---

## Step 1: Install WSL2

### 1.1 Enable WSL2 (PowerShell as Administrator)

```powershell
# Open PowerShell as Administrator, then run:
wsl --install
```

This single command will:
- Enable WSL feature
- Enable Virtual Machine Platform
- Download and install Ubuntu (default)
- Set WSL2 as default

**After installation:**
- Restart your computer when prompted
- Ubuntu will automatically launch after restart
- Create a username and password when prompted (remember these!)

### 1.2 Verify WSL2 Installation

```powershell
# In PowerShell, check WSL version:
wsl --list --verbose

# Should show:
#   NAME      STATE           VERSION
# * Ubuntu    Running         2
```

If VERSION shows 1, upgrade it:
```powershell
wsl --set-version Ubuntu 2
```

---

## Step 2: Update Ubuntu

Open Ubuntu terminal (search "Ubuntu" in Start menu):

```bash
# Update package lists and upgrade existing packages
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential wget git curl
```

---

## Step 3: Install NVIDIA Drivers (Windows side)

**Important**: Install NVIDIA drivers on Windows, NOT in WSL2.

### 3.1 Check Current Driver

```powershell
# In PowerShell:
nvidia-smi
```

You need **NVIDIA Driver 510.06 or newer** for WSL2 CUDA support.

### 3.2 Update if Needed

Download latest driver from: https://www.nvidia.com/Download/index.aspx
- Select your GPU model
- Install on Windows
- Restart if prompted

---

## Step 4: Install CUDA in WSL2

**Important**: Do NOT install NVIDIA drivers in WSL2, only CUDA toolkit!

### 4.1 Verify GPU Access from WSL2

```bash
# In Ubuntu terminal:
nvidia-smi

# Should show your GPU! If not, restart WSL:
# (in PowerShell): wsl --shutdown
# Then reopen Ubuntu terminal
```

### 4.2 Install CUDA 12.6 (Recommended)

```bash
# Download CUDA repository pin
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Add CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/

# Install CUDA
sudo apt update
sudo apt install -y cuda-toolkit-12-6

# Clean up installer
rm cuda-repo-ubuntu2204-12-6-local_12.6.0-560.28.03-1_amd64.deb
```

### 4.3 Set up CUDA environment variables

```bash
# Add to ~/.bashrc
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Reload shell
source ~/.bashrc

# Verify CUDA installation
nvcc --version

# Should show: Cuda compilation tools, release 12.6
```

---

## Step 5: Install Miniconda

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Initialize conda
~/miniconda3/bin/conda init bash

# Reload shell
source ~/.bashrc

# Clean up installer
rm Miniconda3-latest-Linux-x86_64.sh
```

---

## Step 6: Create Mamba Environment

### 6.1 Navigate to Your Repo

```bash
# Your Windows C: drive is mounted at /mnt/c/
cd "/mnt/c/vscode workspace/ml-mangrove"

# Verify you can see your files
ls -la
```

### 6.2 Create Python Environment

```bash
# Create environment with Python 3.10 (compatible with mamba-ssm)
conda create -n mamba-env python=3.10 -y
conda activate mamba-env

# Install PyTorch 2.4.1 with CUDA 12.1 support
# Note: CUDA 12.1 builds work fine with CUDA 12.6 toolkit (backward compatible)
# PyTorch 2.4.1 is the sweet spot - new enough for mamba-ssm APIs, stable C++ ABI
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install NumPy 1.x (required for mamba-ssm compatibility)
pip install "numpy<2.0"

# Update C++ standard library for GLIBCXX_3.4.32 support
conda install -c conda-forge libstdcxx-ng -y

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Should show:
# PyTorch: 2.4.1+cu121
# CUDA available: True
# CUDA version: 12.1
```

---

## Step 7: Install Mamba-SSM

```bash
# Make sure you're in mamba-env
conda activate mamba-env

# Install dependencies
pip install packaging ninja

# IMPORTANT: Build from source to ensure compatibility with PyTorch 2.4.1
# Clear pip cache to avoid using incompatible pre-built wheels
pip cache purge

# Install causal-conv1d from source (required dependency)
pip install causal-conv1d --no-binary causal-conv1d --no-build-isolation

# Install mamba-ssm from source (this will compile CUDA kernels - takes 5-10 minutes!)
pip install mamba-ssm --no-binary mamba-ssm --no-build-isolation

```

**Note**: The `--no-binary` flag forces compilation from source against your specific PyTorch version. This is critical for avoiding C++ ABI compatibility issues.

---

## Step 8: Install Project Dependencies

```bash
# Still in /mnt/c/vscode workspace/ml-mangrove
cd "/mnt/c/vscode workspace/ml-mangrove"

# Install your project requirements
# (adjust path if you have a requirements.txt)
pip install jupyter ipykernel matplotlib pandas scikit-learn albumentations segmentation-models-pytorch

# Install opencv with NumPy 1.x compatibility
pip install opencv-python-headless

# Add kernel to Jupyter
python -m ipykernel install --user --name=mamba-env --display-name="Python (Mamba-WSL2)"
```

---

## Step 9: VSCode Integration

Connect VSCode to WSL2 for seamless development:

1. **Install WSL extension** in VSCode
   - Search for "WSL" in Extensions
   - Install "Remote - WSL" by Microsoft

2. **Open your project in WSL**
   - Press F1 → "WSL: Reopen Folder in WSL"
   - Or click the green button in bottom-left corner

3. **Select kernel**
   - Open `.ipynb` file
   - Click kernel selector (top-right)
   - Choose "Python (Mamba-WSL2)"

---

## Workflow After Setup

### Daily Usage:
1. Open VSCode
2. Click green bottom-left corner → "Reopen in WSL"
3. Open notebook
4. Select "Python (Mamba-WSL2)" kernel
5. Run cells normally

---

## File Access

### From WSL2 → Windows files:
```bash
# Windows C: drive
cd /mnt/c/

# Your repo
cd "/mnt/c/vscode workspace/ml-mangrove"
```

### From Windows → WSL2 files:
```
\\wsl$\Ubuntu\home\<your-username>\
```

**Recommendation**: Keep your repo on Windows (`/mnt/c/`) so both environments can easily access it.

---

## Quick Reference

### Activate environment:
```bash
conda activate mamba-env
```

### Check GPU:
```bash
nvidia-smi
```

### Navigate to project:
```bash
cd "/mnt/c/vscode workspace/ml-mangrove"
```

### Run Jupyter:
```bash
jupyter notebook --no-browser
```

### Stop WSL (from Windows):
```powershell
wsl --shutdown
```
