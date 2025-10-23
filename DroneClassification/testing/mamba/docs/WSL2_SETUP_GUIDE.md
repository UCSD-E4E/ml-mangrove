# WSL2 + CUDA Setup Guide for Mamba Integration

**Purpose**: Set up Linux environment on Windows for training Mamba models while keeping your existing Windows workflow intact.

**Timeline**: ~1-2 hours (depending on download speeds)

---

## Key Version Requirements (Tested & Working)

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

## Step 1: Install WSL2 (15-30 minutes)

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

## Step 2: Update Ubuntu (5-10 minutes)

Open Ubuntu terminal (search "Ubuntu" in Start menu):

```bash
# Update package lists and upgrade existing packages
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential wget git curl
```

---

## Step 3: Install NVIDIA Drivers (Windows side - 10 minutes)

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

## Step 4: Install CUDA in WSL2 (20-30 minutes)

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

## Step 5: Install Miniconda (10 minutes)

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

## Step 6: Create Mamba Environment (15-20 minutes)

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

## Step 7: Install Mamba-SSM (10-15 minutes)

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

# Verify installation
python -c "import mamba_ssm; print('✓ Mamba-SSM installed successfully!')"
```

**Note**: The `--no-binary` flag forces compilation from source against your specific PyTorch version. This is critical for avoiding C++ ABI compatibility issues.

**If this fails**: Check troubleshooting section below.

---

## Step 8: Install Project Dependencies (5 minutes)

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

## Step 9: Test Everything Works (10 minutes)

### 9.1 Test Mamba Installation

```bash
# Navigate to test notebook directory
cd "/mnt/c/vscode workspace/ml-mangrove/DroneClassification/testing/mamba"

# Run test notebook from command line
jupyter nbconvert --to notebook --execute 01_test_mamba.ipynb --output 01_test_mamba_wsl2.ipynb
```

### 9.2 Or use Jupyter in browser

```bash
# Start Jupyter
jupyter notebook --no-browser

# Copy the URL with token (e.g., http://localhost:8888/?token=...)
# Paste in your Windows browser
# Open 01_test_mamba.ipynb
# Select kernel: "Python (Mamba-WSL2)"
# Run all cells
```

---

## Step 10: VSCode Integration (Optional - 5 minutes)

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

**Option A: Terminal-based**
```bash
# In Ubuntu terminal:
cd "/mnt/c/vscode workspace/ml-mangrove"
conda activate mamba-env

# Run training
jupyter nbconvert --to notebook --execute DroneClassification/testing/mamba/04_train_mamba.ipynb
```

**Option B: VSCode (recommended)**
1. Open VSCode
2. Click green bottom-left corner → "Reopen in WSL"
3. Open notebook
4. Select "Python (Mamba-WSL2)" kernel
5. Run cells normally

**Option C: Jupyter in browser**
```bash
# In WSL2:
cd "/mnt/c/vscode workspace/ml-mangrove"
conda activate mamba-env
jupyter notebook --no-browser
# Copy URL to Windows browser
```

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

## Troubleshooting

### Issue: `nvidia-smi` not working in WSL2

**Solution**:
```powershell
# In PowerShell (Windows):
wsl --shutdown
# Wait 10 seconds
# Reopen Ubuntu terminal
```

If still not working:
- Update NVIDIA driver on Windows to 510.06+
- Restart computer

---

### Issue: `mamba-ssm` installation fails

**Error 1**: "undefined symbol" errors (C++ ABI mismatch)
```bash
# This means mamba-ssm was compiled against a different PyTorch version
# Solution: Force rebuild from source
conda activate mamba-env
pip cache purge
pip uninstall -y mamba-ssm causal-conv1d
pip install causal-conv1d --no-binary causal-conv1d --no-build-isolation
pip install mamba-ssm --no-binary mamba-ssm --no-build-isolation
```

**Error 2**: "GLIBCXX_3.4.32 not found"
```bash
# Update C++ standard library
conda activate mamba-env
conda install -c conda-forge libstdcxx-ng -y
```

**Error 3**: NumPy version conflict
```bash
# Downgrade to NumPy 1.x
pip install "numpy<2.0"
# Then rebuild mamba-ssm (see Error 1)
```

**Error 4**: "torch.library has no attribute 'custom_op'"
```bash
# PyTorch version is too old (< 2.4)
# Upgrade to PyTorch 2.4.1:
pip uninstall -y torch torchvision torchaudio
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
# Then rebuild mamba-ssm (see Error 1)
```

**Error 5**: Compilation errors
```bash
# Install build dependencies:
sudo apt install -y build-essential ninja-build
pip install packaging ninja
```

---

### Issue: Jupyter kernel not found

```bash
conda activate mamba-env
python -m ipykernel install --user --name=mamba-env --display-name="Python (Mamba-WSL2)" --force

# Restart Jupyter/VSCode
```

---

### Issue: Out of memory during training

WSL2 uses dynamic memory allocation. To set limits:

Create/edit `C:\Users\<YourUsername>\.wslconfig`:
```ini
[wsl2]
memory=16GB
processors=8
swap=8GB
```

Then restart WSL:
```powershell
wsl --shutdown
```

---

## Performance Tips

### 1. Use Windows filesystem for shared access
- Keep code on `/mnt/c/` (Windows side)
- Slightly slower, but accessible from both environments
- Good for development

### 2. Or use WSL filesystem for max speed
- Clone repo to `~/ml-mangrove` (Linux side)
- Faster I/O during training
- Access from Windows via `\\wsl$\Ubuntu\home\<user>\ml-mangrove`

### 3. Recommended split:
- **Code/notebooks**: Windows filesystem (`/mnt/c/`)
- **Large datasets**: WSL filesystem (`~/data/`)
- **Model checkpoints**: Either (depends on where you need them)

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
