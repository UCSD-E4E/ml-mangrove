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

### Step 5 — Train, evaluate, and build replay buffer (VM, per region)

The full cycle for every region is:
**chip → train → create replay buffer → evaluate → delete chips → upload results to GCS**

> The replay buffer must be created **before** evaluating — `evaluate.py` reads val chips from `/data/replay/<region>/val/`, which doesn't exist until `create_replay_buffer.py` has run.

Val splits are always preserved in `/data/replay/<region>/val/`, so you can re-evaluate any region at any time without re-chipping.

#### Persistent terminal (SSH-safe)

Training takes 10–15 hours per region. Use `tmux` so training keeps running if your SSH connection drops:
```bash
sudo apt-get install -y tmux   # first time only
tmux new -s train              # create a named, persistent terminal session
```
Key tmux commands:
| Action | Keys / Command |
|---|---|
| Detach (leave running in background) | `Ctrl+B` then `D` |
| Reconnect to running session | `tmux attach -t train` |
| List all sessions | `tmux ls` |
| Kill a session | `tmux kill-session -t train` |

If your SSH connection drops mid-training, just SSH back in and `tmux attach -t train` — the process keeps running.

---

#### Region 1 — Florida (base model, no replay)

**Train:**
```bash
python train.py \
    --chips-dir /data/chips \
    --region florida \
    --experiment gee_florida_v1 \
    --epochs 100 \
    --batch-size 8 \
    --num-workers 2 \
    --bucket e4e-mangrove    # change if needed
```

> **Why `--num-workers 2`:** GCP standard persistent disks have a burst IOPS budget that exhausts after ~1 epoch. With 8 workers all reading `.npz` files simultaneously the disk throttles and later epochs slow to a crawl (10× slower). 2 workers is enough to keep the GPU fed without hitting the sustained IOPS limit.

Best checkpoint is saved to `experiments/gee_florida_v1/best_model.pth` and uploaded to GCS automatically on completion.

**Build replay buffer:**
```bash
python scripts/create_replay_buffer.py \
    --chips-dir /data/chips \
    --region florida \
    --replay-dir /data/replay \
    --fraction 0.05
```
Copies ~5% of train chips to `/data/replay/florida/train/` and the full val split to `/data/replay/florida/val/`.

**Evaluate:**
```bash
python evaluate.py \
    --checkpoint "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_florida_v1/best_model.pth" \
    --chips-dir /data/replay \
    --regions florida \
    --output-dir /data/evaluations/florida \
    --batch-size 8
```
No `--baseline` for the first region — nothing to compare against yet.

**Delete chips and upload results:**
```bash
rm -rf /data/chips/florida

gsutil -m cp -r /data/evaluations/florida gs://e4e-mangrove/evaluations/    # change if needed
```

---

#### Region 2 — Brazil (replay: florida)

**Chip + train:**
```bash
python scripts/prepare_chips.py \
    --bucket e4e-mangrove --region brazil \
    --tiles-dir /data/tiles --chips-dir /data/chips

python train.py \
    --chips-dir /data/chips \
    --region brazil \
    --experiment gee_brazil_cl_v1 \
    --resume "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_florida_v1/best_model.pth" \
    --replay-regions florida \
    --replay-dir /data/replay \
    --replay-fraction 0.1 \
    --epochs 100 --batch-size 8 --num-workers 2 \
    --bucket e4e-mangrove    # change if needed
```

**Build replay buffer:**
```bash
python scripts/create_replay_buffer.py \
    --chips-dir /data/chips --region brazil \
    --replay-dir /data/replay --fraction 0.05
```

**Evaluate** (compares Brazil CL model against Florida-only baseline — measures forgetting):
```bash
python evaluate.py \
    --checkpoint "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_brazil_cl_v1/best_model.pth" \
    --chips-dir /data/replay \
    --regions florida brazil \
    --baseline "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_florida_v1/best_model.pth" \
    --output-dir /data/evaluations/brazil_cl \
    --batch-size 8
```
`--baseline` compares the CL checkpoint against the prior single-region model. `forgetting.png` shows ΔIoU per class (red = forgot, green = improved). `summary.json` includes Backward Transfer (BWT) — negative BWT = forgetting, positive = new data helped prior regions.

**Delete chips and upload results:**
```bash
rm -rf /data/chips/brazil

gsutil -m cp -r /data/evaluations/brazil_cl gs://e4e-mangrove/evaluations/    # change if needed
```

---

#### Region 3 — Indonesia (replay: florida, brazil)

**Chip + train:**
```bash
python scripts/prepare_chips.py \
    --bucket e4e-mangrove --region indonesia \
    --tiles-dir /data/tiles --chips-dir /data/chips

python train.py \
    --chips-dir /data/chips \
    --region indonesia \
    --experiment gee_indonesia_cl_v1 \
    --resume "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_brazil_cl_v1/best_model.pth" \
    --replay-regions florida brazil \
    --replay-dir /data/replay \
    --replay-fraction 0.1 \
    --epochs 100 --batch-size 8 --num-workers 2 \
    --bucket e4e-mangrove    # change if needed
```

**Build replay buffer:**
```bash
python scripts/create_replay_buffer.py \
    --chips-dir /data/chips --region indonesia \
    --replay-dir /data/replay --fraction 0.05
```

**Evaluate:**
```bash
python evaluate.py \
    --checkpoint "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_indonesia_cl_v1/best_model.pth" \
    --chips-dir /data/replay \
    --regions florida brazil indonesia \
    --baseline "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_brazil_cl_v1/best_model.pth" \
    --output-dir /data/evaluations/indonesia_cl \
    --batch-size 8
```

**Delete chips and upload results:**
```bash
rm -rf /data/chips/indonesia

gsutil -m cp -r /data/evaluations/indonesia_cl gs://e4e-mangrove/evaluations/    # change if needed
```

---

#### Region 4 — Madagascar/Mozambique (replay: florida, brazil, indonesia)

**Chip + train:**
```bash
python scripts/prepare_chips.py \
    --bucket e4e-mangrove --region madagascar_mozambique \
    --tiles-dir /data/tiles --chips-dir /data/chips

python train.py \
    --chips-dir /data/chips \
    --region madagascar_mozambique \
    --experiment gee_madagascar_cl_v1 \
    --resume "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_indonesia_cl_v1/best_model.pth" \
    --replay-regions florida brazil indonesia \
    --replay-dir /data/replay \
    --replay-fraction 0.1 \
    --epochs 100 --batch-size 8 --num-workers 2 \
    --bucket e4e-mangrove    # change if needed
```

**Build replay buffer:**
```bash
python scripts/create_replay_buffer.py \
    --chips-dir /data/chips --region madagascar_mozambique \
    --replay-dir /data/replay --fraction 0.05
```

**Evaluate:**
```bash
python evaluate.py \
    --checkpoint "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_madagascar_cl_v1/best_model.pth" \
    --chips-dir /data/replay \
    --regions florida brazil indonesia madagascar_mozambique \
    --baseline "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_indonesia_cl_v1/best_model.pth" \
    --output-dir /data/evaluations/madagascar_cl \
    --batch-size 8
```

**Delete chips and upload results:**
```bash
rm -rf /data/chips/madagascar_mozambique

gsutil -m cp -r /data/evaluations/madagascar_cl gs://e4e-mangrove/evaluations/    # change if needed
```

---

#### Region 5 — North Australia (replay: florida, brazil, indonesia, madagascar_mozambique)

**Chip + train:**
```bash
python scripts/prepare_chips.py \
    --bucket e4e-mangrove --region north_australia \
    --tiles-dir /data/tiles --chips-dir /data/chips

python train.py \
    --chips-dir /data/chips \
    --region north_australia \
    --experiment gee_australia_cl_v1 \
    --resume "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_madagascar_cl_v1/best_model.pth" \
    --replay-regions florida brazil indonesia madagascar_mozambique \
    --replay-dir /data/replay \
    --replay-fraction 0.1 \
    --epochs 100 --batch-size 8 --num-workers 2 \
    --bucket e4e-mangrove    # change if needed
```

**Build replay buffer:**
```bash
python scripts/create_replay_buffer.py \
    --chips-dir /data/chips --region north_australia \
    --replay-dir /data/replay --fraction 0.05
```

**Evaluate:**
```bash
python evaluate.py \
    --checkpoint "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_australia_cl_v1/best_model.pth" \
    --chips-dir /data/replay \
    --regions florida brazil indonesia madagascar_mozambique north_australia \
    --baseline "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_madagascar_cl_v1/best_model.pth" \
    --output-dir /data/evaluations/australia_cl \
    --batch-size 8
```

**Delete chips and upload results:**
```bash
rm -rf /data/chips/north_australia

gsutil -m cp -r /data/evaluations/australia_cl gs://e4e-mangrove/evaluations/    # change if needed
```

---

#### Region 6 — East India/Bangladesh (replay: all prior regions)

**Chip + train:**
```bash
python scripts/prepare_chips.py \
    --bucket e4e-mangrove --region east_india_bangladesh \
    --tiles-dir /data/tiles --chips-dir /data/chips

python train.py \
    --chips-dir /data/chips \
    --region east_india_bangladesh \
    --experiment gee_eastindia_cl_v1 \
    --resume "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_australia_cl_v1/best_model.pth" \
    --replay-regions florida brazil indonesia madagascar_mozambique north_australia \
    --replay-dir /data/replay \
    --replay-fraction 0.1 \
    --epochs 100 --batch-size 8 --num-workers 2 \
    --bucket e4e-mangrove    # change if needed
```

**Build replay buffer:**
```bash
python scripts/create_replay_buffer.py \
    --chips-dir /data/chips --region east_india_bangladesh \
    --replay-dir /data/replay --fraction 0.05
```

**Evaluate** (full audit across all 6 regions):
```bash
python evaluate.py \
    --checkpoint "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_eastindia_cl_v1/best_model.pth" \
    --chips-dir /data/replay \
    --regions florida brazil indonesia madagascar_mozambique north_australia east_india_bangladesh \
    --baseline "$HOME/ml-mangrove/Scale Adaption/GEE/experiments/gee_australia_cl_v1/best_model.pth" \
    --output-dir /data/evaluations/final_audit \
    --batch-size 8
```

**Delete chips and upload results:**
```bash
rm -rf /data/chips/east_india_bangladesh

gsutil -m cp -r /data/evaluations/final_audit gs://e4e-mangrove/evaluations/    # change if needed
```

---

#### Key flags (train.py)
| Flag | Default | Description |
|---|---|---|
| `--chips-dir` | required | Root chip directory — has `<region>/train/` and `<region>/val/` |
| `--region` | required | Region to train on |
| `--experiment` | required | Experiment name — creates `experiments/<name>/` |
| `--resume` | None | Checkpoint to warm-start or resume from |
| `--replay-regions` | [] | Prior region names to replay (space-separated) |
| `--replay-dir` | `/data/replay` | Root replay buffer directory |
| `--replay-fraction` | 0.1 | Fraction of each batch sampled from replay |
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 4 | Training batch size |
| `--num-workers` | 4 | DataLoader workers — use 2 on GCP standard PD |
| `--mode` | `full` | Band selection: `full` (68), `rgbn` (4), `rgb` (3), `embeddings` (64) |
| `--lr` | 5e-5 | AdamW learning rate |
| `--weight-decay` | 0.01 | AdamW weight decay |
| `--seed` | 42 | Random seed |
| `--bucket` | None | GCS bucket to upload best checkpoint after training |

### Step 6 — Download results (local machine)

After each region's results are uploaded to GCS, pull them down locally:

```bash
# Download all evaluations
gsutil -m cp -r gs://e4e-mangrove/evaluations/ "Scale Adaption/GEE/evaluations/"    # change bucket if needed

# Download best checkpoints
gsutil -m cp -r gs://e4e-mangrove/checkpoints/ "Scale Adaption/GEE/experiments/"    # change if needed
```

If you get `403 Provided scope(s) are not authorized` on the VM, re-authenticate:
```bash
gcloud auth login --no-launch-browser
gcloud auth application-default login --no-launch-browser
```
Then retry the upload.

#### evaluate.py flags
| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Model checkpoint to evaluate |
| `--chips-dir` | required | Use `/data/replay` — val splits for all regions are here |
| `--regions` | required | One or more region names (space-separated) |
| `--baseline` | None | Prior checkpoint for ΔIoU / BWT forgetting analysis |
| `--split` | `val` | Dataset split to evaluate on (`val` or `train`) |
| `--mode` | `full` | Band mode — must match what was used during training |
| `--patch-size` | 512 | Chip size — must match training |
| `--batch-size` | 4 | Inference batch size |
| `--n-samples` | 6 | Sample prediction images saved per region |

#### Outputs (saved to `--output-dir`)
| File | Description |
|---|---|
| `metrics_<region>.json` | All numerical metrics for that region |
| `confusion_matrix_<region>.png` | Row-normalised confusion matrix |
| `class_metrics_<region>.png` | IoU / Precision / Recall / F1 bar chart |
| `samples_<region>.png` | RGB · Ground Truth · Prediction side-by-side |
| `miou_summary.png` | mIoU + Pixel Accuracy bar chart across all evaluated regions |
| `cl_comparison.png` | Per-class IoU grouped by region (multi-region runs) |
| `forgetting.png` | ΔIoU vs baseline — red = forgetting, green = improvement |
| `summary.json` | All metrics + Backward Transfer (BWT) in one file |

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
