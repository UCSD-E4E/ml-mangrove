# GEE Mangrove Segmentation — Scale Adaptation Pipeline

Multi-region mangrove semantic segmentation using Sentinel-2 + Google Satellite Embeddings via Google Earth Engine. Trains a SegFormer model across 5 geographic regions using continual learning with replay buffers.

---

## Directory Structure

```
Scale Adaption/GEE/
├── config/
│   ├── regions.yaml        # Per-region GEE export config + tile quality filters
│   └── taxonomy.yaml       # Global ESA class mapping (single source of truth)
├── data_pipeline/
│   ├── export_to_gcs.py    # Submit GEE batch export tasks → GCS bucket
│   ├── preflight.py        # Pre-export auth check + bucket state/clear (run locally)
│   └── visualize_tiles.ipynb  # Interactive map + ESA class breakdown before export
├── scripts/
│   ├── vm_setup.sh         # One-time GCP VM environment setup
│   └── prepare_chips.py    # Download tiles from GCS + chip into NPZ (run on VM)
├── gee_dataset.py          # Shared dataset, model, and normalization classes
├── train.py                # Training script — single region or continual learning (run on VM)
├── segformer_training.ipynb   # Florida reference notebook (exploratory)
└── Florida_Training_Dataset/  # Original 102 local Florida tiles (superseded by GCS)
```

---

## Prerequisites

**Local machine:**
- Google Cloud SDK (`gcloud`, `gsutil`)
- Earth Engine Python API (`earthengine-api`, `geemap`)
- Authenticated accounts (see Setup below)

**GCP VM:**
- GPU instance (e.g. `g2-standard-8` + L4, or `n1-standard-8` + V100)
- Run `scripts/vm_setup.sh` once after creation

---

## GCP / EE Account Setup (local)

```bash
gcloud config set account YOUR_EMAIL
gcloud config set project YOUR_GCP_PROJECT_ID   # run: gcloud projects list
gcloud auth application-default login
```

Two separate projects are involved:
| | Project | Purpose |
|---|---|---|
| **GCP** | `YOUR_GCP_PROJECT_ID` | GCS bucket storage, VM compute, billing |
| **EE** | `YOUR_EE_PROJECT` | Earth Engine satellite queries (`ee.Initialize`) |

---

## Full Workflow

### Step 1 — Verify auth + bucket state (local)

```bash
python "Scale Adaption/GEE/data_pipeline/preflight.py" --bucket YOUR_EE_PROJECT --all-regions
```

To clear all region prefixes before a fresh export:
```bash
python "Scale Adaption/GEE/data_pipeline/preflight.py" --bucket YOUR_EE_PROJECT --all-regions --clear
```

### Step 2 — Export all regions to GCS (local)

```bash
python "Scale Adaption/GEE/data_pipeline/export_to_gcs.py" --region florida            --bucket YOUR_EE_PROJECT
python "Scale Adaption/GEE/data_pipeline/export_to_gcs.py" --region brazil             --bucket YOUR_EE_PROJECT
python "Scale Adaption/GEE/data_pipeline/export_to_gcs.py" --region indonesia          --bucket YOUR_EE_PROJECT
python "Scale Adaption/GEE/data_pipeline/export_to_gcs.py" --region madagascar_mozambique --bucket YOUR_EE_PROJECT
python "Scale Adaption/GEE/data_pipeline/export_to_gcs.py" --region east_india_bangladesh --bucket YOUR_EE_PROJECT 
python "Scale Adaption/GEE/data_pipeline/export_to_gcs.py" --region north_australia       --bucket YOUR_EE_PROJECT 
```

Each command submits 100 GEE batch tasks and returns immediately.
Monitor at: https://code.earthengine.google.com/tasks

Export flags:
- `--dry-run` — print tile stats only, no submission
- `--local --limit 2` — download 2 tiles locally to verify before full export

### Step 3 — Set up GCP VM (once)

Create a GPU VM instance via GCP Console (Compute Engine → VM instances → Create Instance).
Recommended machine type: **g2-standard-4** with **1× NVIDIA L4** GPU (CUDA 12.4, 23 GB VRAM).

If you attach a GPU to an existing VM and `nvidia-smi` is not found, install the driver:
```bash
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py
sudo reboot
```

Verify after reboot:
```bash
nvidia-smi   # should show NVIDIA L4
```

SSH into the VM, then bootstrap git and clone the repo first:
```bash
sudo apt-get update && sudo apt-get install -y git
git clone https://github.com/UCSD-E4E/ml-mangrove.git
cd ml-mangrove
```

Then run the full setup script:
```bash
bash "Scale Adaption/GEE/scripts/vm_setup.sh"
```

If conda TOS errors appear mid-script, accept them and re-run:
```bash
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
bash "Scale Adaption/GEE/scripts/vm_setup.sh"
```

After setup completes, initialize conda and activate the environment:
```bash
eval "$($HOME/miniconda3/bin/conda shell.bash hook)" #If haven't already
conda activate mangrove
```

Authenticate with GCS (required once per VM — lets `prepare_chips.py` and `train.py` access the bucket):
```bash
gcloud auth application-default login --no-launch-browser
```
This prints a URL. Open it in your browser, sign in with the GCP account that owns the bucket, copy the verification code, and paste it back in the terminal.

Create the `/data` directory (used for tiles, chips, and experiments):
```bash
sudo mkdir -p /data && sudo chown $USER:$USER /data
```

### Step 4 — Download + chip tiles (VM, per region)

```bash
python scripts/prepare_chips.py \
    --bucket e4e-mangrove \
    --region florida \
    --tiles-dir /data/tiles \
    --chips-dir /data/chips

# Repeat for each region
```

This downloads `.tif` tiles from GCS, computes per-region normalization stats,
and writes 512×512 `.npz` chip files used directly by the training script.
Chips are float16 features + uint8 labels — skips existing files on re-run.

### Step 5 — Train + build replay buffer (VM, per region)

Training takes hours — use `tmux` so it keeps running if your SSH connection drops:
```bash
sudo apt-get install -y tmux   # first time only
tmux new -s train              # create a persistent terminal session
```
Inside the tmux session, run your training command. To detach (leave it running): `Ctrl+B` then `D`.
To reconnect later: `tmux attach -t train`.

**Base model — Florida (no replay):**
```bash
python train.py \
    --chips-dir /data/chips \
    --region florida \
    --experiment gee_florida_v1 \
    --epochs 100 \
    --bucket YOUR_EE_PROJECT
```

**After training each region, sample 5% of chips into a compact replay buffer:**
```bash
python scripts/create_replay_buffer.py \
    --chips-dir /data/chips \
    --region florida \
    --replay-dir /data/replay \
    --fraction 0.05
```

This copies ~5% of chips (~200 chips, ~7 GB) to `/data/replay/florida/train/`. Once the replay buffer is created, you can delete the full chip directory for that region to free space on `/data/`:
```bash
rm -rf /data/chips/florida
```

**Continual learning — each subsequent region:**
```bash
# Chip + train Brazil
python scripts/prepare_chips.py --bucket YOUR_EE_PROJECT --region brazil \
    --tiles-dir /data/tiles --chips-dir /data/chips
python train.py \
    --chips-dir /data/chips \
    --region brazil \
    --experiment gee_brazil_cl_v1 \
    --resume /data/experiments/gee_florida_v1/best_model.pth \
    --replay-regions florida \
    --replay-dir /data/replay \
    --replay-fraction 0.3 \
    --epochs 100 \
    --bucket YOUR_EE_PROJECT
python scripts/create_replay_buffer.py --chips-dir /data/chips --region brazil \
    --replay-dir /data/replay --fraction 0.05
rm -rf /data/chips/brazil

# Indonesia (replay Florida + Brazil)
python scripts/prepare_chips.py --bucket YOUR_EE_PROJECT --region indonesia \
    --tiles-dir /data/tiles --chips-dir /data/chips
python train.py \
    --chips-dir /data/chips \
    --region indonesia \
    --experiment gee_indonesia_cl_v1 \
    --resume /data/experiments/gee_brazil_cl_v1/best_model.pth \
    --replay-regions florida brazil \
    --replay-dir /data/replay \
    --replay-fraction 0.3 \
    --epochs 100 \
    --bucket YOUR_EE_PROJECT
python scripts/create_replay_buffer.py --chips-dir /data/chips --region indonesia \
    --replay-dir /data/replay --fraction 0.05
rm -rf /data/chips/indonesia

# Continue pattern: madagascar_mozambique → north_australia → east_india_bangladesh
# Each run adds the previous region(s) to --replay-regions
```

Best checkpoints are uploaded to `gs://YOUR_EE_PROJECT/checkpoints/<experiment>/best_model.pth`.

---

## Configuration

### regions.yaml

Per-region tile selection config. Key fields:
- `bbox` or `country_filter` / `multi_country` — geographic extent
- `date_range` — Sentinel-2 composite window
- `min_mangrove_pct` / `max_water_pct` / `max_tree_pct` / `max_bare_built_pct` — quality filters
- `sample_size` — number of tiles to export (100 per region)

| Region | Coverage | Key imbalance |
|---|---|---|
| `florida` | South Florida + Cuba + Bahamas | Water-heavy coastal tiles |
| `brazil` | Brazil | Amazon inland forest floods candidate pool |
| `indonesia` | Indonesia (bbox) | Archipelago = many ocean-dominated tiles |
| `madagascar_mozambique` | Madagascar + Mozambique | Estuarine channel tiles |
| `north_australia` | Kimberley + Top End + Gulf of Carpentaria | Savanna woodland + tidal mudflats |
| `east_india_bangladesh` | Sundarbans bbox | Dense inland forest + high cropland |

### taxonomy.yaml

Global 9-class ESA mapping used by all scripts. Decided from the all-regions ESA breakdown:

| Class | ESA | FL | Brazil | Indonesia | Madagascar | East India |
|---|---|---|---|---|---|---|
| Trees | 10 | 24.6% | 32.4% | 36.4% | 18.5% | 13.9% |
| Shrubland | 20 | 0.4% | 1.3% | 0.7% | 3.6% | ~0% |
| Grassland | 30 | 15.7% | 15.6% | 5.0% | 30.8% | 1.3% |
| Cropland | 40 | 2.9% | 1.2% | 5.1% | 2.5% | **20.6%** |
| Built-up | 50 | 0.7% | 2.9% | 0.4% | 0.4% | 0.1% |
| Bare/Sparse | 60 | 0.1% | 0.5% | 0.2% | 1.6% | 0.6% |
| Water | 80 | 34.6% | 27.8% | 40.9% | 25.9% | 37.8% |
| Wetland | 90 | 10.2% | 7.4% | 0.7% | 5.4% | ~0% |
| Mangrove | 95 | 10.8% | 10.9% | 10.6% | 11.4% | 25.6% |

Snow/Ice (70) and Moss/Lichen (100) excluded — 0% across all regions.
Classes absent in a region are `IGNORE_INDEX=255` during training (no gradient, no noise).

---

## Tile Format (69-band GeoTIFF)

| Bands | Content |
|---|---|
| 1–4 | Sentinel-2 RGBN optical (raw reflectance ×10000) |
| 5–68 | Google Satellite Embeddings (64-d float32, DINO-based) |
| 69 | ESA WorldCover v200 label (raw values: 10, 20, …, 95) |

Resolution: 10m/px · Tile size: 2048×2048 · CRS: EPSG:3857

---

## Continual Learning Design

**Why replay buffers?**
Sequential fine-tuning on new regions causes catastrophic forgetting — the model overwrites Florida-learned weights with Brazil gradients. Replaying a fraction of previous chips in each batch prevents this.

**Why a fixed 9-class head from the start?**
Classes rare in Florida (e.g. Cropland) are `IGNORE_INDEX` there — their logits simply receive no gradient. When East India arrives (20.6% Cropland), those logits start training naturally. No architectural changes between regions.

**Replay fraction:** 0.3 (30% of each batch from prior regions) is a reasonable starting point.
Too low → forgetting. Too high → new region learns slowly. Tune per experiment.

**Training order (recommended):** Florida → Brazil → Indonesia → Madagascar → East India
