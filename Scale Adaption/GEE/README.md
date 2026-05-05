# GEE Mangrove Segmentation — Scale Adaptation Pipeline

Multi-region mangrove semantic segmentation using Sentinel-2 + Google Satellite Embeddings via Google Earth Engine. Trains a SegFormer model across 6 geographic regions using continual learning with replay buffers.

---

## Directory Structure

```
Scale Adaption/GEE/
├── config/
│   ├── regions.yaml           # Per-region GEE export config + tile quality filters
│   └── taxonomy.yaml          # Global ESA class mapping (single source of truth)
├── data_pipeline/
│   ├── export_to_gcs.py       # Submit GEE batch export tasks → GCS bucket
│   ├── preflight.py           # Pre-export auth check + bucket state/clear (run locally)
│   └── visualize_tiles.ipynb  # Interactive map + ESA class breakdown before export
├── scripts/
│   ├── vm_setup.sh            # One-time GCP VM environment setup
│   ├── prepare_chips.py       # Download tiles from GCS + chip into NPZ (run on VM)
│   └── create_replay_buffer.py  # Sample train chips + copy full val into replay dir
├── gee_dataset.py             # Shared dataset, model, and normalization classes
├── train.py                   # Training script — single region or continual learning
├── evaluate.py                # Evaluation — per-region, forgetting check, full CL audit
├── segformer_training.ipynb   # Florida reference notebook (exploratory, not updated)
└── Florida_Training_Dataset/  # Original 102 local Florida tiles (superseded by GCS)
```

---

## Prerequisites

**Local machine:**
- Google Cloud SDK (`gcloud`, `gsutil`)
- Earth Engine Python API (`earthengine-api`, `geemap`)
- Authenticated accounts (see Setup below)

**GCP VM:**
- GPU instance — recommended: **g2-standard-12** with **1× NVIDIA L4** (23 GB VRAM, CUDA 12.4)
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
python "Scale Adaption/GEE/data_pipeline/preflight.py" \
    --bucket e4e-mangrove \    # change if needed
    --all-regions
```

To clear all region prefixes before a fresh export:
```bash
python "Scale Adaption/GEE/data_pipeline/preflight.py" \
    --bucket e4e-mangrove \    # change if needed
    --all-regions --clear
```

### Step 2 — Export all regions to GCS (local)

```bash
python "Scale Adaption/GEE/data_pipeline/export_to_gcs.py" --region florida               --bucket e4e-mangrove  # change if needed
python "Scale Adaption/GEE/data_pipeline/export_to_gcs.py" --region brazil                --bucket e4e-mangrove  # change if needed
python "Scale Adaption/GEE/data_pipeline/export_to_gcs.py" --region indonesia             --bucket e4e-mangrove  # change if needed
python "Scale Adaption/GEE/data_pipeline/export_to_gcs.py" --region madagascar_mozambique --bucket e4e-mangrove  # change if needed
python "Scale Adaption/GEE/data_pipeline/export_to_gcs.py" --region east_india_bangladesh --bucket e4e-mangrove  # change if needed
python "Scale Adaption/GEE/data_pipeline/export_to_gcs.py" --region north_australia       --bucket e4e-mangrove  # change if needed
```

Each command submits 100 GEE batch tasks and returns immediately.
Monitor at: https://code.earthengine.google.com/tasks

Export flags:
- `--dry-run` — print tile stats only, no submission
- `--local --limit 2` — download 2 tiles locally to verify before full export

### Step 3 — Set up GCP VM (once)

Create a GPU VM instance via GCP Console (Compute Engine → VM instances → Create Instance).
Recommended machine type: **g2-standard-12** with **1× NVIDIA L4** GPU (CUDA 12.4, 23 GB VRAM).

If `nvidia-smi` is not found after VM creation, install the kernel headers first (DKMS needs them to build the driver modules), then the driver, then reboot:
```bash
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r)
sudo apt-get install -y nvidia-driver-550-open
sudo reboot
```

Verify after reboot:
```bash
nvidia-smi   # should show NVIDIA L4
```

After confirming the GPU is visible, reinstall PyTorch with CUDA support:
```bash
conda activate mangrove
pip install "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected: True / NVIDIA L4
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
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"   # if not already active
conda activate mangrove
```

Authenticate with GCS (required once per VM — lets `prepare_chips.py` and `train.py` access the bucket):
```bash
gcloud auth login --no-launch-browser
gcloud auth application-default login --no-launch-browser
```
Both commands print a URL. Open each in your browser, sign in with the GCP account that owns the bucket, copy the verification code, and paste it back in the terminal.

Create the `/data` directory (used for tiles, chips, replay buffers, and evaluations):
```bash
sudo mkdir -p /data && sudo chown $USER:$USER /data
```

### Step 4 — Download + chip tiles (VM, per region)

```bash
python scripts/prepare_chips.py \
    --bucket e4e-mangrove \      # change if needed
    --region florida \
    --tiles-dir /data/tiles \
    --chips-dir /data/chips

# Repeat for each region: brazil, indonesia, madagascar_mozambique, north_australia, east_india_bangladesh
```

Key flags:
- `--bucket` — GCS bucket name
- `--region` — must match a key in `config/regions.yaml`
- `--tiles-dir` — where raw `.tif` tiles are downloaded
- `--chips-dir` — where `.npz` chip files are written
- `--mode` — band selection: `full` (68 bands, default), `rgbn` (4), `rgb` (3), `embeddings` (64)
- `--patch-size` — chip size in pixels (default: 512)
- `--stride-train` / `--stride-val` — sliding window stride (defaults: 256 / 512)
- `--val-fraction` — fraction of tiles held out for validation (default: 0.2)

This downloads `.tif` tiles from GCS, computes per-region normalization stats,
and writes 512×512 `.npz` chip files. Chips are float16 features + uint8 labels — skips existing files on re-run.

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
    --batch-size 8 \
    --num-workers 8 \
    --bucket e4e-mangrove    # change if needed
```

Key flags:
- `--chips-dir` — root chip directory (has `<region>/train/` and `<region>/val/`)
- `--region` — region to train on
- `--experiment` — experiment name (creates `experiments/<name>/`)
- `--resume` — path to checkpoint to warm-start or resume from
- `--replay-regions` — list of prior region names to replay
- `--replay-dir` — root replay buffer directory (default: `/data/replay`)
- `--replay-fraction` — fraction of each batch from replay (default: 0.3)
- `--mode` — band mode matching the chips (default: `full`)
- `--lr` — learning rate (default: 5e-5)
- `--weight-decay` — AdamW weight decay (default: 0.01)
- `--seed` — random seed (default: 42)
- `--bucket` — GCS bucket to upload best checkpoint after training

**After training each region, build the replay buffer (keeps 5% train + full val):**
```bash
python scripts/create_replay_buffer.py \
    --chips-dir /data/chips \
    --region florida \
    --replay-dir /data/replay \
    --fraction 0.05
```

This copies ~5% (~196) of train chips to `/data/replay/florida/train/` and the full val split to `/data/replay/florida/val/`. Once done, delete the full chip directory to free space:
```bash
rm -rf /data/chips/florida
```

**Continual learning — each subsequent region:**
```bash
# Brazil (replay Florida)
python scripts/prepare_chips.py --bucket e4e-mangrove --region brazil \    # change if needed
    --tiles-dir /data/tiles --chips-dir /data/chips
python train.py \
    --chips-dir /data/chips \
    --region brazil \
    --experiment gee_brazil_cl_v1 \
    --resume "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_florida_v1/best_model.pth" \
    --replay-regions florida \
    --replay-dir /data/replay \
    --replay-fraction 0.1 \
    --epochs 100 \
    --batch-size 8 \
    --num-workers 8 \
    --bucket e4e-mangrove    # change if needed
python scripts/create_replay_buffer.py --chips-dir /data/chips --region brazil \
    --replay-dir /data/replay --fraction 0.05
rm -rf /data/chips/brazil

# Indonesia (replay Florida + Brazil)
python scripts/prepare_chips.py --bucket e4e-mangrove --region indonesia \    # change if needed
    --tiles-dir /data/tiles --chips-dir /data/chips
python train.py \
    --chips-dir /data/chips \
    --region indonesia \
    --experiment gee_indonesia_cl_v1 \
    --resume "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_brazil_cl_v1/best_model.pth" \
    --replay-regions florida brazil \
    --replay-dir /data/replay \
    --replay-fraction 0.1 \
    --epochs 100 \
    --batch-size 8 \
    --num-workers 8 \
    --bucket e4e-mangrove    # change if needed
python scripts/create_replay_buffer.py --chips-dir /data/chips --region indonesia \
    --replay-dir /data/replay --fraction 0.05
rm -rf /data/chips/indonesia

# Continue pattern: madagascar_mozambique → north_australia → east_india_bangladesh
# Each run: add the previous region to --replay-regions and update --resume to the latest checkpoint
```

Best checkpoints are uploaded to `gs://e4e-mangrove/checkpoints/<experiment>/best_model.pth`.  # change if needed

### Step 6 — Evaluate (VM)

`evaluate.py` covers three scenarios: current region, forgetting check, and full CL audit.

Checkpoints are saved inside the repo under `Scale Adaption/GEE/experiments/`. The val split for each region is preserved in `/data/replay/<region>/val/` — use `--chips-dir /data/replay` when evaluating prior regions so you don't need to re-chip them.

**After Florida — evaluate current region:**
```bash
python evaluate.py \
    --checkpoint "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_florida_v1/best_model.pth" \
    --chips-dir /data/chips \
    --regions florida \
    --output-dir /data/evaluations/florida
```

**After Brazil CL — check forgetting on Florida + current on Brazil:**
```bash
python evaluate.py \
    --checkpoint "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_brazil_cl_v1/best_model.pth" \
    --chips-dir /data/replay \
    --regions florida brazil \
    --baseline "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_florida_v1/best_model.pth" \
    --output-dir /data/evaluations/brazil_cl_check
```

**Full audit across all trained regions:**
```bash
python evaluate.py \
    --checkpoint "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_indonesia_cl_v1/best_model.pth" \
    --chips-dir /data/replay \
    --regions florida brazil indonesia \
    --output-dir /data/evaluations/indonesia_full
```

Key flags:
- `--checkpoint` — model to evaluate
- `--chips-dir` — use `/data/chips` for the current region, `/data/replay` for prior regions
- `--regions` — one or more region names
- `--baseline` — optional prior checkpoint for forgetting/BWT analysis
- `--split` — `val` (default) or `train`
- `--mode` — band mode matching the chips (default: `full`)
- `--patch-size` — must match the value used during training (default: 512)
- `--batch-size` — inference batch size (default: 4)
- `--n-samples` — number of sample prediction images per region (default: 6)

**Outputs per run** (saved to `--output-dir`):
| File | Description |
|---|---|
| `metrics_<region>.json` | All numerical metrics per region |
| `confusion_matrix_<region>.png` | Row-normalized confusion matrix |
| `class_metrics_<region>.png` | IoU / Precision / Recall / F1 bar chart |
| `samples_<region>.png` | RGB · Ground Truth · Prediction side-by-side |
| `miou_summary.png` | mIoU + Pixel Accuracy across all regions |
| `cl_comparison.png` | Per-class IoU grouped by region (multi-region runs) |
| `forgetting.png` | ΔIoU vs baseline — red=forgetting, green=improvement |
| `summary.json` | All metrics + Backward Transfer (BWT) in one file |

**Copy results to GCS (from VM):**
```bash
gsutil -m cp -r /data/evaluations/ gs://e4e-mangrove/evaluations/    # change if needed
```

**Download results to your local machine (run locally):**
```bash
# Download all evaluations
gsutil -m cp -r gs://e4e-mangrove/evaluations/ "Scale Adaption/GEE/evaluations/"    # change if needed

# Download best checkpoints
gsutil -m cp -r gs://e4e-mangrove/checkpoints/ "Scale Adaption/GEE/experiments/"    # change if needed
```

If you get `403 Provided scope(s) are not authorized` on the VM, re-authenticate:
```bash
gcloud auth login --no-launch-browser
gcloud auth application-default login --no-launch-browser
```
Then retry the upload.

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
Classes absent in a region receive no gradient (their pixels never appear as labels).

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
Classes rare in Florida (e.g. Cropland) simply never appear as pixels there — their logits receive no gradient. When East India arrives (20.6% Cropland), those logits start training naturally. No architectural changes between regions.

**Replay fraction:** 0.1 (10% of each batch from prior regions) is a good balance across 6 regions.
Too low → forgetting. Too high → new region learns slowly. Tune per experiment.

**Training order (recommended):** Florida → Brazil → Indonesia → Madagascar/Mozambique → North Australia → East India/Bangladesh
