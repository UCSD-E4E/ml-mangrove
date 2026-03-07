# End-to-End Examples

This guide provides complete, runnable examples for common workflows. Each example builds on concepts from the previous guides, showing how all the pieces fit together in practice.

---

## Example 1: Complete Training Pipeline

This example walks through the entire process from raw GeoTIFF imagery to a trained model, step by step.

### The Scenario

You have drone imagery of a coastal area stored as GeoTIFF files, along with shapefile annotations marking mangrove regions. Your goal is to train a model that can automatically detect mangroves in new imagery.

### Step 1: Organize Your Raw Data

First, create the expected folder structure and place your data in it:

```python
from DroneClassification.data.utils import make_chunk_dirs
import os

# Define paths
BASE_PATH = "data/mangrove_project"
RAW_DATA_PATH = os.path.join(BASE_PATH, "raw")
PROCESSED_PATH = os.path.join(BASE_PATH, "processed")

# Create folder structure for 10 chunks (adjust based on your data)
make_chunk_dirs(RAW_DATA_PATH, num_chunks=10)
```

After running this, manually place your files:
- Each GeoTIFF goes in its chunk folder (e.g., `raw/Chunks/Chunk 1/Chunk1.tif`)
- Shapefiles go in the `labels/` subfolder (e.g., `raw/Chunks/Chunk 1/labels/annotations.shp`)

Your folder structure should look like:

```
data/mangrove_project/
└── raw/
    └── Chunks/
        ├── Chunk 1/
        │   ├── Chunk1.tif           # Your RGB GeoTIFF
        │   └── labels/
        │       ├── annotations.shp   # Shapefile
        │       ├── annotations.shx
        │       ├── annotations.dbf
        │       └── annotations.prj
        ├── Chunk 2/
        │   ├── Chunk2.tif
        │   └── labels/
        │       └── ...
        └── ...
```

### Step 2: Convert Shapefiles to Raster Labels

Your shapefiles contain vector polygons, but neural networks need pixel-wise labels. Convert them:

```python
from DroneClassification.data.utils import rasterize_shapefiles

# Convert all shapefiles to label TIFFs
rasterize_shapefiles(os.path.join(RAW_DATA_PATH, "Chunks"))

# This creates a labels.tif file in each chunk folder
```

**What to check:** After this runs, each chunk folder should have a `labels.tif` file. The console output will tell you if any chunks had issues (missing shapefiles, invalid geometries, etc.).

### Step 3: Create Training Tiles

Now cut the large GeoTIFFs into small tiles and save them as memory-mapped arrays:

```python
from DroneClassification.data.utils import tile_dataset
import os

# Configuration
TILE_SIZE = 224  # Standard size for pretrained models
FILTER_RATIO = 0.8  # Discard 80% of single-class tiles

# Output paths
IMAGES_FILE = os.path.join(PROCESSED_PATH, "images.npy")
LABELS_FILE = os.path.join(PROCESSED_PATH, "labels.npy")

# Create output directory
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Tile the dataset
tile_dataset(
    data_path=os.path.join(RAW_DATA_PATH, "Chunks"),
    combined_images_file=IMAGES_FILE,
    combined_labels_file=LABELS_FILE,
    image_size=TILE_SIZE,
    filter_monolithic_labels=FILTER_RATIO
)
```

**What to expect:** This will take a while for large datasets. You'll see progress for each chunk, and at the end it reports the total number of tiles created.

### Step 4: Explore and Validate Your Data

Before training, always check that your data looks correct:

```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the data (memory-mapped, so this is fast)
images = np.load(IMAGES_FILE, mmap_mode='r')
labels = np.load(LABELS_FILE, mmap_mode='r')

print(f"Dataset size: {len(images)} tiles")
print(f"Image shape: {images[0].shape}")  # Should be (3, 224, 224)
print(f"Label shape: {labels[0].shape}")  # Should be (1, 224, 224)

# Check label distribution
sample_labels = labels[:1000]  # Sample first 1000
unique_values = np.unique(sample_labels)
print(f"Unique label values: {unique_values}")  # Should be [0, 1, 255] typically

# Visualize a few samples
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for i in range(3):
    idx = np.random.randint(len(images))

    # Original image
    img = images[idx].transpose(1, 2, 0)  # CHW -> HWC
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"Image {idx}")
    axes[i, 0].axis('off')

    # Label mask
    label = labels[idx].squeeze()
    axes[i, 1].imshow(label, cmap='tab10', vmin=0, vmax=2)
    axes[i, 1].set_title(f"Label (unique: {np.unique(label)})")
    axes[i, 1].axis('off')

    # Overlay
    overlay = img.copy().astype(float) / 255
    mask = (label == 1)  # Mangrove pixels
    overlay[mask] = overlay[mask] * 0.5 + np.array([0, 1, 0]) * 0.5  # Green tint
    axes[i, 2].imshow(overlay)
    axes[i, 2].set_title("Overlay")
    axes[i, 2].axis('off')

    # Class distribution
    values, counts = np.unique(label, return_counts=True)
    axes[i, 3].bar([str(v) for v in values], counts)
    axes[i, 3].set_title("Class distribution")

plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_PATH, "data_preview.png"))
plt.show()
```

**What to look for:**
- Images should show actual imagery (not blank or corrupted)
- Labels should have reasonable class distributions (not all one class)
- Overlays should show labels aligned with visible features

### Step 5: Prepare Datasets and Data Loaders

Split your data into training and validation sets:

```python
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from DroneClassification.data.MemoryMapDataset import MemmapDataset

# Define transforms
train_transforms = T.Compose([
    T.RandomHorizontalFlip(0.5),
    T.RandomVerticalFlip(0.5),
    T.RandomRotation(degrees=15),
])

valid_transforms = T.Compose([])  # No augmentation for validation

# Load dataset
dataset = MemmapDataset(
    images=IMAGES_FILE,
    labels=LABELS_FILE
)

# Shuffle the dataset (modifies the files in-place)
print("Shuffling dataset...")
dataset.shuffle(flush_interval=1000)

# Split into train/validation (80/20)
train_dataset, valid_dataset = dataset.split(
    split_ratio=0.8,
    valid_transforms=valid_transforms
)

# Apply training transforms to train set
train_dataset.transforms = train_transforms

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")

# Create data loaders
BATCH_SIZE = 32
NUM_WORKERS = 4  # Adjust based on your CPU cores

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True  # Speeds up GPU transfer
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# Verify a batch loads correctly
batch_images, batch_labels = next(iter(train_loader))
print(f"Batch images shape: {batch_images.shape}")  # Should be [32, 3, 224, 224]
print(f"Batch labels shape: {batch_labels.shape}")  # Should be [32, 1, 224, 224]
```

### Step 6: Calculate Class Weights (Optional but Recommended)

If your data is imbalanced (usually is), calculate weights:

```python
from DroneClassification.training_utils import calculate_class_weights

classes = {0: "Background", 1: "Mangrove"}
weights = calculate_class_weights(LABELS_FILE, classes, power=2.0)

# Output example:
# Class distribution:
#   Background     : 12,345,678 pixels (85.3%)
#   Mangrove       :  2,134,567 pixels (14.7%)
#
# Normalized class weights:
#   Background     : 0.3456
#   Mangrove       : 1.6544

print(f"Mangrove class weight: {weights[1].item():.4f}")
```

### Step 7: Create Model and Loss Function

Choose your model architecture and loss function:

```python
from DroneClassification.models import DeepLab, JaccardLoss
from DroneClassification.training_utils import setup_device

# Setup device
device = setup_device()

# Create model
model = DeepLab(
    num_classes=1,           # Binary segmentation
    input_image_size=224,
    backbone='resnet50',
    output_stride=8          # Higher resolution features
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create loss function
loss_fn = JaccardLoss(
    num_classes=1,
    ignore_index=255,        # Ignore unknown pixels
    alpha=0.5,               # Balance between CE and Jaccard
    boundary_weight=0.1      # Add boundary awareness
)
```

### Step 8: Train the Model

Now put it all together and train:

```python
from DroneClassification.training_utils import TrainingSession

# Create training session
session = TrainingSession(
    model=model,
    trainLoader=train_loader,
    testLoader=valid_loader,
    lossFunc=loss_fn,
    num_epochs=100,
    init_lr=0.001,
    weight_decay=1e-4,
    experiment_name="mangrove_deeplab_v1",
    class_names=["Background", "Mangrove"],
    metric_mode="boundary",      # Track boundary IoU
    save_checkpoints=True
)

# Start training
session.learn()

# Training runs and produces:
# - Progress bars for each epoch
# - Validation metrics after each epoch
# - Loss and metric plots at the end
# - Best model saved to experiments/mangrove_deeplab_v1/best_model.pth
```

**What to watch during training:**
- Training loss should decrease over time
- Validation IoU should increase, then plateau
- If validation loss increases while training loss decreases, you're overfitting

### Step 9: Evaluate Results

After training, examine the results:

```python
from DroneClassification.training_utils import visualize_segmentation_results
import numpy as np

# Load raw images for visualization
raw_images = np.load(IMAGES_FILE, mmap_mode='r')

# Visualize predictions on validation samples
for sample_idx in [0, 50, 100, 200]:
    visualize_segmentation_results(
        model=session.model,
        dataset=valid_dataset,
        raw_images=raw_images,
        sample_idx=sample_idx,
        class_names=["Background", "Mangrove"],
        device=device
    )
```

### Step 10: Save Final Model for Deployment

The best model is automatically saved, but you can also export it explicitly:

```python
import torch

# Save just the model weights (smallest file, for inference)
torch.save(
    session.model.state_dict(),
    os.path.join(PROCESSED_PATH, "mangrove_model_final.pth")
)

print("Model saved!")
print(f"Best validation IoU: {session.best_metric:.4f} (epoch {session.best_epoch + 1})")
```

---

## Example 2: Running Inference on New Imagery

Once you have a trained model, here's how to use it on new GeoTIFF imagery.

### Loading Your Trained Model

```python
import torch
import numpy as np
from PIL import Image
from DroneClassification.models import DeepLab
from DroneClassification.training_utils import setup_device

# Setup
device = setup_device()
TILE_SIZE = 224  # Must match what you trained with
MODEL_PATH = "experiments/mangrove_deeplab_v1/best_model.pth"

# Recreate model architecture (must match training)
model = DeepLab(
    num_classes=1,
    input_image_size=TILE_SIZE,
    backbone='resnet50',
    output_stride=8
)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print("Model loaded successfully!")
```

### Inference on a Single Tile

```python
def predict_tile(model, image_array, device):
    """
    Run inference on a single image tile.

    Args:
        model: Trained PyTorch model
        image_array: numpy array, shape (H, W, 3), values 0-255
        device: torch device

    Returns:
        Binary mask, shape (H, W), values 0 or 1
    """
    # Normalize using ImageNet statistics
    image = image_array.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    # Convert to tensor: (H, W, C) -> (1, C, H, W)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    image_tensor = image_tensor.to(device)

    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.sigmoid(output)
        mask = (probabilities > 0.5).squeeze().cpu().numpy()

    return mask.astype(np.uint8)

# Example usage
test_image = np.array(Image.open("test_tile.png"))
prediction = predict_tile(model, test_image, device)

# Visualize
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(test_image)
axes[0].set_title("Input")
axes[1].imshow(prediction, cmap='Greens')
axes[1].set_title("Prediction")
overlay = test_image.copy().astype(float) / 255
overlay[prediction == 1] = overlay[prediction == 1] * 0.5 + np.array([0, 1, 0]) * 0.5
axes[2].imshow(overlay)
axes[2].set_title("Overlay")
plt.show()
```

### Inference on a Large GeoTIFF (Sliding Window)

For images larger than your tile size, use a sliding window approach:

```python
import rasterio
from tqdm import tqdm

def predict_large_geotiff(model, geotiff_path, output_path, tile_size=224, device='cuda'):
    """
    Run inference on a large GeoTIFF using sliding window.

    Args:
        model: Trained PyTorch model
        geotiff_path: Path to input GeoTIFF
        output_path: Path to save prediction GeoTIFF
        tile_size: Size of tiles to process
        device: torch device

    Returns:
        Full prediction mask as numpy array
    """
    with rasterio.open(geotiff_path) as src:
        # Read image data
        image = src.read()  # Shape: (C, H, W)
        image = image[:3]   # Use only RGB channels
        image = image.transpose(1, 2, 0)  # (H, W, C)

        height, width = image.shape[:2]
        meta = src.meta.copy()

    # Create output array
    full_prediction = np.zeros((height, width), dtype=np.uint8)
    count_array = np.zeros((height, width), dtype=np.uint8)  # For averaging overlaps

    # Calculate number of tiles
    n_rows = (height + tile_size - 1) // tile_size
    n_cols = (width + tile_size - 1) // tile_size

    print(f"Processing {n_rows * n_cols} tiles...")

    model.eval()
    with torch.no_grad():
        for row in tqdm(range(n_rows)):
            for col in range(n_cols):
                # Calculate tile boundaries
                y_start = row * tile_size
                x_start = col * tile_size
                y_end = min(y_start + tile_size, height)
                x_end = min(x_start + tile_size, width)

                # Extract tile
                tile = image[y_start:y_end, x_start:x_end]

                # Pad if necessary (edge tiles might be smaller)
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    padded = np.zeros((tile_size, tile_size, 3), dtype=tile.dtype)
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded

                # Run prediction
                pred = predict_tile(model, tile, device)

                # Crop prediction back to original tile size
                pred = pred[:y_end - y_start, :x_end - x_start]

                # Accumulate predictions
                full_prediction[y_start:y_end, x_start:x_end] += pred
                count_array[y_start:y_end, x_start:x_end] += 1

    # Average overlapping predictions (if any)
    full_prediction = (full_prediction / np.maximum(count_array, 1)).astype(np.uint8)

    # Save as GeoTIFF with same georeferencing
    meta.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(full_prediction, 1)

    print(f"Prediction saved to {output_path}")
    return full_prediction

# Usage
prediction = predict_large_geotiff(
    model=model,
    geotiff_path="new_imagery/site_A.tif",
    output_path="predictions/site_A_prediction.tif",
    tile_size=224,
    device=device
)
```

---

## Example 3: Model-Assisted Labeling with CVAT

This workflow uses model predictions to speed up manual annotation. Instead of labeling from scratch, annotators refine model predictions.

### Overview

1. Tile your GeoTIFF into smaller images
2. Generate initial predictions with your model
3. Export predictions in CVAT format
4. Import into CVAT for refinement
5. Export corrected annotations
6. Use corrected data for retraining

### Step 1: Tile the GeoTIFF

```python
import os
import numpy as np
from PIL import Image
import rasterio
from pathlib import Path

def tile_geotiff_for_annotation(geotiff_path, output_dir, tile_size=512):
    """
    Cut a GeoTIFF into tiles for annotation.

    Args:
        geotiff_path: Path to input GeoTIFF
        output_dir: Directory to save tiles
        tile_size: Size of each tile

    Returns:
        List of tile paths and their coordinates
    """
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(geotiff_path) as src:
        image = src.read()[:3].transpose(1, 2, 0)  # RGB, HWC format
        height, width = image.shape[:2]

    tile_info = []

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            # Extract tile
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            tile = image[y:y_end, x:x_end]

            # Skip mostly empty tiles
            if tile.mean() < 5:  # Very dark = likely no data
                continue

            # Pad if needed
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded

            # Save tile
            tile_name = f"tile_{y:05d}_{x:05d}.png"
            tile_path = os.path.join(output_dir, tile_name)
            Image.fromarray(tile).save(tile_path)

            tile_info.append({
                'path': tile_path,
                'y': y,
                'x': x,
                'y_end': y_end,
                'x_end': x_end
            })

    print(f"Created {len(tile_info)} tiles in {output_dir}")
    return tile_info

# Usage
WORKSPACE = Path("mal_workspace")
TILES_DIR = WORKSPACE / "01_tiles"

tile_info = tile_geotiff_for_annotation(
    geotiff_path="new_imagery/area_to_annotate.tif",
    output_dir=str(TILES_DIR),
    tile_size=512
)
```

### Step 2: Generate Predictions

```python
from tqdm import tqdm

PREDICTIONS_DIR = WORKSPACE / "02_predictions"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

print(f"Generating predictions for {len(tile_info)} tiles...")

for info in tqdm(tile_info):
    # Load tile
    tile = np.array(Image.open(info['path']))

    # Predict
    pred_mask = predict_tile(model, tile, device)

    # Save prediction
    tile_name = Path(info['path']).stem
    pred_path = PREDICTIONS_DIR / f"{tile_name}_pred.png"
    Image.fromarray(pred_mask).save(pred_path)

print(f"Predictions saved to {PREDICTIONS_DIR}")
```

### Step 3: Create CVAT Export Package

```python
import zipfile
import tempfile

def create_cvat_export(predictions_dir, output_zip, class_names, class_colors=None):
    """
    Create a CVAT-compatible segmentation mask ZIP.

    CVAT expects this structure:
        archive.zip/
        ├── labelmap.txt
        ├── ImageSets/Segmentation/default.txt
        ├── SegmentationClass/
        │   └── *.png (class masks)
        └── SegmentationObject/
            └── *.png (same as SegmentationClass for semantic seg)
    """
    if class_colors is None:
        class_colors = {
            0: (0, 0, 0),        # background - black
            1: (0, 128, 0),      # mangrove - green
        }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create directories
        (temp_path / "SegmentationClass").mkdir()
        (temp_path / "SegmentationObject").mkdir()
        (temp_path / "ImageSets" / "Segmentation").mkdir(parents=True)

        # Create labelmap.txt
        with open(temp_path / "labelmap.txt", 'w') as f:
            for class_id, name in enumerate(class_names):
                color = class_colors.get(class_id, (0, 0, 0))
                f.write(f"{name}:{color[0]},{color[1]},{color[2]}::\n")

        # Copy prediction masks and build file list
        pred_files = sorted(predictions_dir.glob("*_pred.png"))
        image_names = []

        for pred_file in tqdm(pred_files, desc="Preparing export"):
            base_name = pred_file.stem.replace("_pred", "")
            image_names.append(base_name)

            # Copy to both directories (same for semantic segmentation)
            mask = np.array(Image.open(pred_file))
            Image.fromarray(mask).save(temp_path / "SegmentationClass" / f"{base_name}.png")
            Image.fromarray(mask).save(temp_path / "SegmentationObject" / f"{base_name}.png")

        # Create default.txt
        with open(temp_path / "ImageSets" / "Segmentation" / "default.txt", 'w') as f:
            for name in image_names:
                f.write(name + '\n')

        # Create ZIP
        os.makedirs(os.path.dirname(output_zip), exist_ok=True)
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_path):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(temp_path)
                    zipf.write(file_path, arcname)

        print(f"CVAT export created: {output_zip}")
        print(f"  - {len(image_names)} masks")
        print(f"  - {len(class_names)} classes")

# Create export
create_cvat_export(
    predictions_dir=PREDICTIONS_DIR,
    output_zip=str(WORKSPACE / "cvat_predictions.zip"),
    class_names=["background", "mangrove"],
    class_colors={0: (0, 0, 0), 1: (0, 128, 0)}
)
```

### Step 4: Use CVAT for Refinement

Now use the CVAT web interface:

1. **Start CVAT** (if using Docker):
   ```bash
   cd cvat
   docker compose up -d
   ```
   Open http://localhost:8080

2. **Create a project** with labels matching your class names

3. **Create a task** and upload the tiles from `mal_workspace/01_tiles/`

4. **Import predictions**:
   - Menu → Upload annotations
   - Select "Segmentation mask 1.1" format
   - Upload `mal_workspace/cvat_predictions.zip`

5. **Refine annotations** using brush and eraser tools

6. **Export corrected masks**:
   - Menu → Export task dataset
   - Select "Segmentation mask 1.1" format
   - Extract to `mal_workspace/03_corrected_masks/`

### Step 5: Reconstruct Full GeoTIFF from Corrected Tiles

```python
def reconstruct_geotiff(tile_info, masks_dir, output_path, original_geotiff):
    """
    Reconstruct a full GeoTIFF from corrected tile masks.
    """
    with rasterio.open(original_geotiff) as src:
        meta = src.meta.copy()
        height, width = src.height, src.width

    # Create full mask
    full_mask = np.zeros((height, width), dtype=np.uint8)

    for info in tqdm(tile_info, desc="Reconstructing"):
        tile_name = Path(info['path']).stem
        mask_path = masks_dir / f"{tile_name}.png"

        if not mask_path.exists():
            continue

        mask = np.array(Image.open(mask_path))

        # Handle size differences
        y, x = info['y'], info['x']
        y_end, x_end = info['y_end'], info['x_end']
        mask_h, mask_w = y_end - y, x_end - x

        full_mask[y:y_end, x:x_end] = mask[:mask_h, :mask_w]

    # Save with original georeferencing
    meta.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(full_mask, 1)

    print(f"Reconstructed mask saved to {output_path}")

# Usage
reconstruct_geotiff(
    tile_info=tile_info,
    masks_dir=WORKSPACE / "03_corrected_masks" / "SegmentationClass",
    output_path=str(WORKSPACE / "corrected_labels.tif"),
    original_geotiff="new_imagery/area_to_annotate.tif"
)
```

---

## Example 4: Comparing Multiple Models

When experimenting with different architectures, it's helpful to compare them systematically.

```python
from DroneClassification.models import ResNet_UNet, DeepLab, SegFormer
from DroneClassification.training_utils import TrainingSession, plot_comparison_metrics

# Define models to compare
model_configs = [
    {
        'name': 'ResNet-UNet',
        'model': ResNet_UNet(input_image_size=224, num_classes=1),
    },
    {
        'name': 'DeepLab-ResNet50',
        'model': DeepLab(num_classes=1, input_image_size=224, backbone='resnet50'),
    },
    {
        'name': 'SegFormer-B2',
        'model': SegFormer(num_classes=1, input_image_size=224),
    },
]

# Train each model
all_metrics = []
model_names = []

for config in model_configs:
    print(f"\n{'='*60}")
    print(f"Training: {config['name']}")
    print('='*60)

    session = TrainingSession(
        model=config['model'],
        trainLoader=train_loader,
        testLoader=valid_loader,
        lossFunc=loss_fn,
        num_epochs=50,
        init_lr=0.001,
        experiment_name=f"comparison_{config['name'].replace('-', '_').lower()}",
        save_checkpoints=True
    )

    session.learn()

    all_metrics.append(session.metrics)
    model_names.append(config['name'])

# Compare final metrics
final_metrics = [[m[-1]] for m in all_metrics]  # Last epoch metrics

plot_comparison_metrics(
    title="Model Comparison - Final Metrics",
    metrics=final_metrics,
    titles=model_names,
    metrics_wanted=['Precision', 'Recall', 'IoU']
)
```

---

## Tips and Best Practices

### Memory Management

For large datasets, be mindful of memory:

```python
# Use memory-mapped arrays
images = np.load("images.npy", mmap_mode='r')  # Read-only, stays on disk

# Process in batches for analysis
batch_size = 1000
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    # Process batch...
```

### Reproducibility

Set random seeds for reproducible experiments:

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

### Debugging Training Issues

If training isn't progressing:

```python
# 1. Check data loading
batch = next(iter(train_loader))
print(f"Images: {batch[0].shape}, {batch[0].min():.3f} to {batch[0].max():.3f}")
print(f"Labels: {batch[1].shape}, unique: {batch[1].unique()}")

# 2. Check model output
model.eval()
with torch.no_grad():
    output = model(batch[0].to(device))
    print(f"Output: {output.shape}, {output.min():.3f} to {output.max():.3f}")

# 3. Check loss
loss = loss_fn(output, batch[1].to(device))
print(f"Loss: {loss.item():.4f}")

# 4. Check gradients
model.train()
output = model(batch[0].to(device))
loss = loss_fn(output, batch[1].to(device))
loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad mean={param.grad.mean():.6f}, max={param.grad.max():.6f}")
```

### Monitoring GPU Usage

```python
import torch

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

    # Clear cache if needed
    torch.cuda.empty_cache()
```

---

## Quick Reference

| Task | Example Location |
|------|------------------|
| Full training pipeline | Example 1 |
| Inference on single tile | Example 2, `predict_tile()` |
| Inference on large GeoTIFF | Example 2, `predict_large_geotiff()` |
| Model-assisted labeling | Example 3 |
| Comparing models | Example 4 |
| CVAT export creation | Example 3, `create_cvat_export()` |

---

## Existing Notebooks

The repository includes several notebooks with working examples:

| Notebook | Description |
|----------|-------------|
| `data/process_data.ipynb` | Data preparation workflow |
| `model_training_ground.ipynb` | Training experiments |
| `testing/mamba/04_train_deeplab.ipynb` | DeepLab training example |
| `testing/mamba/05b_cvat_simplified_workflow.ipynb` | CVAT integration workflow |

These notebooks contain real working code you can adapt for your own projects.
