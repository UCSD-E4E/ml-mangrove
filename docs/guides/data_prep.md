# Data Preparation Guide

This guide explains how to prepare geospatial imagery for training mangrove segmentation models. We'll walk through the entire pipeline from raw satellite or drone imagery to training-ready datasets.

---

## Understanding the Problem

When working with satellite or drone imagery for machine learning, there is one fundamental challenge that makes it quite different from other computer vision task. The images that we're dealing with are enormous. A single drone survey might produce a GeoTIFF file that's several gigabytes in size, containing millions of pixels. Neural networks, however, work with small fixed-size images—typically 224×224 or 512×512 pixels.

The data preparation pipeline solves this by:

1. **Breaking large images into tiles** - Cutting your massive GeoTIFF into thousands of small, consistent tiles that neural networks can process
2. **Converting labels to the right format** - Your annotations might be in vector format (shapefiles from QGIS or CVAT), but neural networks need pixel-wise labels as raster images
3. **Storing data efficiently** - Using memory-mapped files so you can work with datasets larger than your computer's RAM
4. **Preparing images for pretrained models** - Normalizing pixel values to match what models like ResNet or SegFormer expect

---

## The Big Picture

Here's what happens to your data as it flows through the pipeline:

```
Your Raw Data                         What the Model Sees
─────────────                         ───────────────────

Large GeoTIFF                         Thousands of small tiles
(e.g., 20000×15000 pixels)    →      (e.g., 224×224 pixels each)

Vector shapefiles                     Pixel-wise label masks
(polygons drawn in QGIS)      →      (same size as image tiles)
```

The pipeline handles all of this automatically. You provide the raw imagery and annotations, and it produces training-ready numpy arrays.

---

## Where Everything Lives

All data preparation code is in `DroneClassification/data/`:

- **utils.py** - The heavy lifting: reading GeoTIFFs, converting shapefiles to rasters, cutting images into tiles, and resampling resolutions
- **MemoryMapDataset.py** - A PyTorch Dataset class that loads your prepared data efficiently during training
- **normalization_pipeline.py** - Utilities for normalizing pixel values (usually handled automatically)

---

## Step 1: Organizing Your Raw Data

Before any processing can happen, your data needs to be organized in a specific folder structure. The pipeline expects your imagery to be split into "chunks"—this is useful because you often have multiple drone flights or satellite scenes that cover different areas.

Use `make_chunk_dirs()` to create this structure:

```python
from DroneClassification.data.utils import make_chunk_dirs

make_chunk_dirs("path/to/data", num_chunks=5)
```

This creates a folder structure like:

```
data/
└── Chunks/
    ├── Chunk 1/
    │   └── labels/
    ├── Chunk 2/
    │   └── labels/
    ├── Chunk 3/
    │   └── labels/
    └── ...
```

**After creating this structure, you need to manually:**
- Place each GeoTIFF image in its corresponding chunk folder (named like `Chunk1.tif`, `Chunk2.tif`)
- Place the shapefile annotations in each chunk's `labels/` subfolder

Why chunks? This organization helps when you have data from different sources, different dates, or different geographic areas. It also makes processing more manageable since each chunk is handled independently before everything is combined at the end.

---

## Step 2: Converting Shapefiles to Raster Labels

If you've annotated your imagery using tools like QGIS, CVAT, or ArcGIS, your labels are probably stored as shapefiles—vector data that defines polygons around mangrove areas. However, for semantic segmentation, the model needs labels as a raster image where each pixel has a class value.

The `rasterize_shapefiles()` function handles this conversion:

```python
from DroneClassification.data.utils import rasterize_shapefiles

rasterize_shapefiles("path/to/data/Chunks")
```

**What this does:**
- Goes through each chunk folder looking for shapefiles in the `labels/` subfolder
- Reads the corresponding GeoTIFF to get the exact dimensions and georeferencing
- Burns the vector polygons into a raster image with matching dimensions
- Saves the result as `labels.tif` in each chunk folder

**How labels are encoded:**
- `1` = mangrove (or your positive class)
- `0` = non-mangrove (background)
- `255` = ignore/unknown (areas that should be excluded from training)

The function automatically looks for common column names in your shapefiles (`label`, `labels`, `class`, or `value`) and converts text labels like "mangrove" and "non-mangrove" to numeric values.

If your shapefiles have invalid geometries (self-intersections, etc.), the function attempts to repair them automatically. You'll see messages in the console if any issues are found and fixed.

---

## Step 3: Creating Training Tiles

Now comes the core transformation: cutting your large GeoTIFFs into small, uniform tiles that neural networks can process. The `tile_dataset()` function does this and saves everything as memory-mapped numpy arrays.

```python
from DroneClassification.data.utils import tile_dataset

tile_dataset(
    data_path="path/to/data/Chunks",
    combined_images_file="path/to/output/images.npy",
    combined_labels_file="path/to/output/labels.npy",
    image_size=224,
    filter_monolithic_labels=0.8
)
```

**What this does:**
- Iterates through every chunk, reading both the image GeoTIFF and labels GeoTIFF
- Slides a window across each image, extracting tiles of the specified size (default 224×224)
- Pairs each image tile with its corresponding label tile
- Optionally filters out uninformative tiles (see below)
- Combines all tiles from all chunks into two large numpy arrays
- Saves these arrays as memory-mapped files for efficient access

**About the tile size:** The default of 224×224 pixels matches what ImageNet-pretrained models expect. If your imagery has very high resolution (like 5cm/pixel drone imagery), you might want larger tiles to capture enough context. If it's coarser satellite imagery, 224 might be just right.

**About filtering monolithic labels:** In any large dataset, many tiles will be entirely one class—maybe a tile over open water (all non-mangrove) or deep in a mangrove forest (all mangrove). While these aren't useless, having too many of them can slow down training without adding much learning signal. The `filter_monolithic_labels` parameter randomly discards a percentage of these single-class tiles. Setting it to `0.8` means 80% of tiles that are entirely one class will be discarded, keeping 20% of them. Set to `None` or `0` to keep everything.

---

## Step 4: Loading Data for Training

Once your data is processed into numpy arrays, you need to load it into PyTorch for training. The `MemmapDataset` class handles this, with a key feature: it uses memory-mapped files, meaning it doesn't load the entire dataset into RAM. Instead, it reads data directly from disk as needed.

This is critical for large datasets. If you have 100,000 training tiles at 224×224×3 pixels, that's roughly 15GB of data. Without memory mapping, you'd need that much RAM just to load your dataset—before even running the model!

```python
from DroneClassification.data.MemoryMapDataset import MemmapDataset

dataset = MemmapDataset(
    images="path/to/images.npy",
    labels="path/to/labels.npy"
)
```

**What you get back:**
- A standard PyTorch Dataset that works with DataLoader
- Each item is a tuple of (image_tensor, label_tensor)
- Images are automatically converted to float tensors and scaled to [0, 1]
- ImageNet normalization is applied (subtracting mean, dividing by std)—this is important for pretrained models

### Splitting into Training and Validation

Machine learning best practice is to hold out some data for validation—to check how well your model generalizes to data it hasn't seen during training. The dataset provides two ways to do this:

**Option 1: In-memory split (faster, no disk writes)**
```python
train_dataset, valid_dataset = dataset.split(
    split_ratio=0.8,
    valid_transforms=valid_transforms
)
```

This creates two "views" of the same underlying data. The first 80% becomes training data, the last 20% becomes validation. No data is copied.

**Option 2: Save split to disk (useful for reproducibility)**
```python
dataset.save_training_split(
    output_dir="path/to/output",
    data_split=0.9
)
```

This physically copies the data into separate files:
- `output/train/images.npy` and `output/train/labels.npy`
- `output/valid/images.npy` and `output/valid/labels.npy`

### Shuffling

Before training, you typically want to shuffle your data so the model sees examples in random order. The dataset provides an in-place shuffle:

```python
dataset.shuffle(flush_interval=1000)
```

**Important:** This actually modifies the numpy files on disk using the Fisher-Yates shuffle algorithm. The `flush_interval` parameter controls how often changes are written to disk—useful on Windows systems which can struggle with memory management on large files.

### Combining Multiple Datasets

If you have data from different sources (different field campaigns, different sensors), you can combine them:

```python
combined = dataset1.concat(dataset2, output_path="path/to/combined")
```

This creates new numpy files containing all tiles from both datasets. The original files are not modified.

### Applying Data Augmentation

You can pass custom transforms to the dataset for data augmentation:

```python
import torchvision.transforms.v2 as v2

train_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomHorizontalFlip(0.5),
    v2.RandomVerticalFlip(0.5),
    v2.RandomRotation(degrees=15),
])

dataset = MemmapDataset(
    images="path/to/images.npy",
    labels="path/to/labels.npy",
    transforms=train_transforms
)
```

Augmentation artificially increases your dataset size by randomly transforming images during training—flipping, rotating, changing brightness, etc. This helps the model generalize better.

---

## Optional: Resampling Resolution

Sometimes your imagery isn't at the resolution you need. Maybe you have very high-resolution drone imagery (2cm/pixel) but want to train at a coarser resolution to match satellite data you'll eventually use for inference.

The `resample()` function reprojects your data to a target resolution:

```python
from DroneClassification.data.utils import resample

resample(
    input_root="path/to/high_res_data",
    output_root="path/to/resampled_data",
    target_resolution=0.5,
    save_backup_labels=True
)
```

**What this does:**
- Reads each GeoTIFF and calculates the appropriate UTM projection for its location
- Resamples to the target resolution (in meters per pixel)
- Realigns the shapefile labels to match the new pixel grid
- Applies smoothing to prevent jagged edges in the resampled labels

The `save_backup_labels` option keeps a copy of your original label files in case you need them later.

---

## Normalization (Usually Automatic)

The `Normalizer` class in `normalization_pipeline.py` provides utilities for normalizing pixel values. However, you usually don't need to call these directly—`MemmapDataset` handles ImageNet normalization automatically.

**Why normalization matters:** Pretrained models like ResNet were trained on ImageNet, where images were normalized to have specific mean and standard deviation values per channel. If you feed in raw pixel values (0-255), the model's pretrained weights won't work well. The normalization ensures your mangrove imagery looks statistically similar to what the model saw during pretraining.

If you do need manual normalization:

```python
from DroneClassification.data.normalization_pipeline import Normalizer

# ImageNet normalization (for pretrained models)
normalized = Normalizer.normalize_imagenet(image_array)
original = Normalizer.denormalize_imagenet(normalized)

# Min-max normalization (scales to [0, 1])
normalized, min_val, max_val = Normalizer.normalize_minmax(image_array)
original = Normalizer.denormalize_minmax(normalized, min_val, max_val)
```

---

## Putting It All Together

Here's a complete workflow from raw data to training-ready DataLoaders:

```python
from DroneClassification.data.utils import (
    make_chunk_dirs,
    rasterize_shapefiles,
    tile_dataset
)
from DroneClassification.data.MemoryMapDataset import MemmapDataset
from torch.utils.data import DataLoader

# 1. Create folder structure
make_chunk_dirs("data/raw", num_chunks=10)

# 2. (Manually place your GeoTIFFs and shapefiles in the chunk folders)

# 3. Convert vector labels to raster format
rasterize_shapefiles("data/raw/Chunks")

# 4. Tile everything into training-ready arrays
tile_dataset(
    data_path="data/raw/Chunks",
    combined_images_file="data/processed/images.npy",
    combined_labels_file="data/processed/labels.npy",
    image_size=224,
    filter_monolithic_labels=0.7
)

# 5. Load as a PyTorch dataset
dataset = MemmapDataset(
    images="data/processed/images.npy",
    labels="data/processed/labels.npy"
)

# 6. Shuffle and split
dataset.shuffle()
train_ds, valid_ds = dataset.split(0.8, valid_transforms)

# 7. Create DataLoaders for training
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_ds, batch_size=32, num_workers=4)
```

---

## Quick Reference

| What you want to do | Function/Class |
|---------------------|----------------|
| Create the folder structure for raw data | `make_chunk_dirs()` |
| Convert shapefile annotations to raster labels | `rasterize_shapefiles()` |
| Cut large images into small training tiles | `tile_dataset()` |
| Change the resolution of your imagery | `resample()` |
| Load prepared data for PyTorch training | `MemmapDataset` |
| Split data into train/validation sets | `dataset.split()` or `dataset.save_training_split()` |
| Shuffle data randomly | `dataset.shuffle()` |
| Combine multiple datasets | `dataset.concat()` |

---

## Next Steps

Once your data is prepared, you're ready to train a model. See:
- Model Training *(coming soon)* - How to train segmentation models
- Inference *(coming soon)* - How to run predictions on new imagery
