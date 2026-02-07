# Training Guide

This guide explains how to train segmentation models using the training utilities in this repository. We'll cover the complete training workflow, from setting up your training session to monitoring progress and saving your best models.

---

## The Training Loop Explained

Training a neural network is fundamentally about repetition and gradual improvement. The process works like this:

1. **Show the model some images** (a "batch") and have it make predictions
2. **Calculate how wrong it was** using a loss function
3. **Adjust the model's weights** slightly to reduce that error (backpropagation)
4. **Repeat** thousands of times across all your training data

One complete pass through all your training data is called an "epoch". You typically train for many epochs,maybe 50, 100, or more, until the model stops improving.

The training utilities in this repository handle all of this automatically. The `TrainingSession` class manages the training loop, tracks metrics, saves checkpoints, and produces visualizations so you can monitor progress.

---

## Where Everything Lives

All training utilities are in `DroneClassification/training_utils/`:

- **training_utils.py** - The main `TrainingSession` class for running training, plus helper functions like `setup_device()` and `calculate_class_weights()`
- **SegmentationDataset.py** - An alternative PyTorch Dataset that loads data into RAM (faster iteration, but requires enough memory)
- **data_visualization.py** - Functions for plotting results, comparing models, and visualizing predictions

You can import everything directly:

```python
from DroneClassification.training_utils import (
    TrainingSession,
    setup_device,
    calculate_class_weights,
    SegmentationDataset,
    visualize_segmentation_results
)
```

---

## Setting Up Your Training Environment

### Automatic Device Selection

Before training, you need to decide whether to run on CPU or GPU. The `setup_device()` function handles this automatically:

```python
from DroneClassification.training_utils import setup_device

device = setup_device()
# Prints: "Using CUDA device." or "Using Apple Metal Performance Shaders (MPS) device."
#         or "WARNING: No GPU found. Defaulting to CPU."
```

**What it checks (in order):**
1. CUDA (NVIDIA GPUs) - The fastest option for most setups
2. MPS (Apple Silicon) - For M1/M2/M3 Macs
3. CPU - Fallback if no GPU is available

Training on GPU is dramatically faster—often 10-50x faster than CPU. If you have an NVIDIA GPU, make sure you have CUDA properly installed.

---

## The TrainingSession Class

The `TrainingSession` class is the heart of the training framework. It wraps your model, data, and training configuration into a single object that handles the entire training process.

### Basic Usage

```python
from DroneClassification.training_utils import TrainingSession
from DroneClassification.models import ResNet_UNet, JaccardLoss
from torch.utils.data import DataLoader

# Create your model and loss
model = ResNet_UNet(input_image_size=224, num_classes=1)
loss_fn = JaccardLoss(num_classes=1, ignore_index=255)

# Create data loaders (see Data Preparation guide)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)

# Create training session
session = TrainingSession(
    model=model,
    trainLoader=train_loader,
    testLoader=valid_loader,
    lossFunc=loss_fn,
    num_epochs=50,
    init_lr=0.001
)

# Start training
session.learn()
```

That's it! The `learn()` method runs the complete training loop, including validation after each epoch, metric tracking, checkpoint saving, and generating loss/metric plots at the end.

### Understanding the Parameters

Let's break down what each parameter does:

**Required parameters:**

| Parameter | Description |
|-----------|-------------|
| `model` | Your neural network (any PyTorch `nn.Module`) |
| `trainLoader` | DataLoader for training data |
| `testLoader` | DataLoader for validation data (can be a list for multiple validation sets) |
| `lossFunc` | Loss function to optimize |

**Training configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 10 | How many times to iterate through all training data |
| `init_lr` | 0.001 | Initial learning rate (how big the weight updates are) |
| `weight_decay` | 1e-4 | L2 regularization strength (prevents overfitting) |
| `threshold` | 0.5 | Probability threshold for binary predictions |
| `ignore_index` | 255 | Label value to ignore in loss/metrics (typically "unknown") |

**Customization:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimizer` | AdamW | Custom optimizer (if None, uses AdamW) |
| `scheduler` | CosineAnnealing | Learning rate scheduler (if None, uses cosine annealing) |
| `device` | Auto-detected | Manually specify CPU/GPU |
| `class_names` | None | List of class names for better logging |

**Experiment tracking:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `experiment_name` | Timestamped | Name for the experiment directory |
| `save_checkpoints` | True | Whether to save model checkpoints |
| `epoch_print_frequency` | 1 | How often to print detailed metrics |

---

### What Happens During Training

When you call `session.learn()`, here's what happens:

1. **For each epoch:**
   - The model is set to training mode
   - Each batch of training data is passed through the model
   - Loss is calculated and backpropagated
   - Weights are updated
   - Progress bar shows current loss and learning rate

2. **After each epoch:**
   - The model is switched to evaluation mode
   - Validation data is passed through (without gradient tracking)
   - Metrics are calculated (IoU, precision, recall, pixel accuracy)
   - Results are logged to console and file
   - If this is the best model so far, it's saved

3. **After all epochs:**
   - Loss curves are plotted
   - Metric curves are plotted
   - Training summary is logged

### Experiment Directory Structure

When `save_checkpoints=True`, TrainingSession creates an organized experiment directory:

```
experiments/
└── experiment_20240115_143022/    # Or your custom experiment_name
    ├── config.json                # Training configuration
    ├── training.log               # Detailed training logs
    ├── best_model.pth             # Weights from best epoch
    ├── latest_checkpoint.pth      # Complete checkpoint (weights + optimizer + scheduler)
    ├── epoch_10_checkpoint.pth    # Periodic checkpoints (every 10 epochs)
    ├── training_loss.png          # Loss curve visualization
    └── metrics.png                # Metric curves visualization
```

**The difference between `best_model.pth` and checkpoints:**
- `best_model.pth` contains only the model weights—use this for inference
- Checkpoint files contain everything needed to resume training (weights, optimizer state, scheduler state, metrics history)

---

### Validation Metrics

The training session automatically calculates these metrics during validation:

| Metric | What it Measures |
|--------|------------------|
| `Loss` | The loss function value (lower is better) |
| `Pixel_Accuracy` | Percentage of pixels correctly classified |
| `Precision` | Of pixels predicted as mangrove, how many actually are |
| `Recall` | Of actual mangrove pixels, how many did we find |
| `IoU` | Intersection over Union (the most important metric for segmentation) |
| `Boundary_IoU` | IoU specifically for boundary pixels (how accurate are the edges) |

**Understanding these metrics:**

- **High precision, low recall**: The model is conservative—when it says "mangrove", it's usually right, but it misses a lot of actual mangroves
- **Low precision, high recall**: The model is aggressive—it finds most mangroves but also has many false positives
- **IoU** balances both: it's the standard metric for segmentation and penalizes both false positives and false negatives

For multi-class segmentation, you'll also get per-class metrics: `class_ious`, `class_precisions`, `class_recalls`.

### Metric Modes

You can configure how metrics are calculated using `metric_mode`:

```python
session = TrainingSession(
    ...,
    metric_mode="segmentation"  # Default
)
```

Available modes:
- `"segmentation"` - Standard segmentation metrics (IoU, precision, recall, etc.)
- `"boundary"` - Segmentation metrics plus boundary IoU
- `"chip"` - For classification tasks (just accuracy)
- `"none"` - Only compute loss (fastest, useful for debugging)

You can also register custom metric functions:

```python
def my_custom_metrics(predictions, targets):
    # Your metric calculation here
    return {"custom_metric": value}

session.register_metric_fn("custom", my_custom_metrics)
session.metric_mode = "custom"
```

---

### Multiple Validation Sets

Sometimes you want to validate on multiple datasets—for example, to see how your model performs on different geographic regions or image types. TrainingSession supports this:

```python
valid_loader_region_a = DataLoader(dataset_region_a, batch_size=32)
valid_loader_region_b = DataLoader(dataset_region_b, batch_size=32)

session = TrainingSession(
    ...,
    testLoader=[valid_loader_region_a, valid_loader_region_b],
    validation_dataset_names=["Region A", "Region B"]
)
```

Metrics will be computed and logged separately for each validation set, helping you understand if your model generalizes well across different data sources.

---

### Learning Rate Scheduling

Learning rate scheduling is crucial for good training. The learning rate controls how big the weight updates are—too high and training is unstable, too low and it's slow.

By default, TrainingSession uses **cosine annealing**: the learning rate starts at `init_lr` and gradually decreases following a cosine curve to near zero by the end of training. This is a good default that works well for most problems.

If you want different scheduling:

```python
import torch.optim as optim

# Create your own scheduler
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',      # Reduce LR when metric stops improving
    factor=0.5,      # Multiply LR by 0.5
    patience=5       # Wait 5 epochs before reducing
)

session = TrainingSession(
    ...,
    optimizer=optimizer,
    scheduler=scheduler
)
```

**Common scheduler choices:**
- `CosineAnnealingLR` - Smooth decrease, good default
- `ReduceLROnPlateau` - Reduces LR when progress stalls
- `StepLR` - Reduces LR by fixed factor every N epochs
- `OneCycleLR` - Increases then decreases LR (can train faster)

---

### Mixed Precision Training

TrainingSession automatically uses mixed precision training when running on CUDA or CPU. This means some computations are done in 16-bit floating point instead of 32-bit, which:

- Uses less GPU memory (you can use larger batches)
- Runs faster (modern GPUs are optimized for 16-bit)
- Maintains accuracy (critical operations stay in 32-bit)

You don't need to configure anything—it's automatic. On MPS (Apple Silicon), mixed precision is disabled due to compatibility issues.

---

### Resuming Training

If training is interrupted, you can resume from a checkpoint:

```python
session = TrainingSession(...)

# Load the checkpoint
epoch, metrics = session.load_checkpoint("experiments/my_experiment/latest_checkpoint.pth")
print(f"Resuming from epoch {epoch}")

# Continue training (you might want to adjust num_epochs)
session.learn()
```

If you just want to load model weights (for inference or fine-tuning):

```python
session.load_model_weights("experiments/my_experiment/best_model.pth")
```

---

## Handling Class Imbalance

In most mangrove datasets, there are far more "not mangrove" pixels than "mangrove" pixels. This class imbalance can cause the model to simply predict "not mangrove" everywhere and still achieve high pixel accuracy.

### Calculating Class Weights

The `calculate_class_weights()` function analyzes your labels and computes weights that make the loss function pay more attention to rare classes:

```python
from DroneClassification.training_utils import calculate_class_weights

# Define your classes
classes = {0: "Background", 1: "Mangrove"}

# Calculate weights from your labels file
weights = calculate_class_weights(
    labels_path="data/labels.npy",
    classes=classes,
    power=2.0  # Higher = more aggressive rebalancing
)

# Output:
# Class distribution:
#   Background     : 12,345,678 pixels (92.3%)
#   Mangrove       :  1,030,456 pixels ( 7.7%)
#
# Normalized class weights:
#   Background     : 0.1663
#   Mangrove       : 1.8337
```

The `power` parameter controls how aggressive the rebalancing is:
- `power=1.0` - Weights are proportional to inverse frequency (mild)
- `power=2.0` - Weights are proportional to inverse frequency squared (default, stronger)
- `power=3.0` - Even more aggressive rebalancing

**Using the weights:**

Many loss functions accept weights:

```python
from DroneClassification.models import FocalLoss

loss_fn = FocalLoss(alpha=weights[1].item())  # Use mangrove class weight
```

Or for multi-class with Cross-Entropy:

```python
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss(weight=weights)
```

### Multi-Class and Multi-Channel Labels

`calculate_class_weights()` handles various label formats:

**Single-channel labels (most common):**
```python
# Labels shape: (N, H, W) with values 0, 1, 2, ...
classes = {0: "Background", 1: "Mangrove", 2: "Water"}
```

**Multi-channel one-hot labels:**
```python
# Labels shape: (N, C, H, W) where C = number of classes
classes = {0: "Background", 1: "Mangrove", 2: "Water"}
```

**Multi-channel combination labels:**
```python
# Labels shape: (N, 2, H, W) where class is determined by channel combination
classes = {
    (0, 0): "Background",
    (1, 0): "Mangrove",
    (0, 1): "Water",
    (1, 1): "Mangrove_Water"  # Overlapping
}
```

---

## Alternative Dataset: SegmentationDataset

While `MemmapDataset` (from the data module) reads data directly from disk, `SegmentationDataset` loads everything into RAM. This is faster during training but requires enough memory to hold your entire dataset.

```python
from DroneClassification.training_utils import SegmentationDataset
import torchvision.transforms.v2 as T

transforms = T.Compose([
    T.RandomHorizontalFlip(0.5),
    T.RandomVerticalFlip(0.5),
])

dataset = SegmentationDataset(
    images="data/images.npy",
    labels="data/labels.npy",
    transforms=transforms,
    load_to_ram=True  # Set to False to keep as memory-mapped
)
```

**When to use SegmentationDataset:**
- Your dataset fits comfortably in RAM (maybe under 8GB)
- You want maximum iteration speed
- You're doing lots of augmentation (avoids repeated disk reads)

**When to use MemmapDataset instead:**
- Large datasets that don't fit in RAM
- You're memory-constrained
- You only need to iterate once or twice (like for evaluation)

Both work with `TrainingSession`—just wrap them in a DataLoader.

---

## Visualizing Results

### Segmentation Visualization

After training, you'll want to see what your model actually produces. The `visualize_segmentation_results()` function creates a comprehensive visualization:

```python
from DroneClassification.training_utils import visualize_segmentation_results

visualize_segmentation_results(
    model=session.model,
    dataset=valid_dataset,
    raw_images=np.load("data/images.npy", mmap_mode='r'),
    sample_idx=42,  # Which sample to visualize
    class_names=["Background", "Mangrove"],
    device=device
)
```

This generates a 6-panel figure showing:
1. **Original image** - The input RGB image
2. **Ground truth** - The actual labels (color-coded by class)
3. **Prediction** - What the model predicted
4. **Difference map** - Where predictions differ from ground truth
5. **Ground truth distribution** - Bar chart of class frequencies in the label
6. **Prediction distribution** - Bar chart of class frequencies in the prediction

Plus detailed console output with pixel counts, accuracy metrics, and a qualitative assessment.

### Comparing Models

If you've trained multiple models and want to compare them:

```python
from DroneClassification.training_utils import plot_comparison_metrics

# metrics1 and metrics2 are the metrics lists from TrainingSession.metrics
plot_comparison_metrics(
    title="Model Comparison",
    metrics=[metrics1, metrics2],
    titles=["ResNet-UNet", "SegFormer"],
    metrics_wanted=['Precision', 'Recall', 'IoU']
)
```

For comparing across different resolutions or configurations:

```python
from DroneClassification.training_utils import compare_series_by_resolution

compare_series_by_resolution(
    title="Performance vs Resolution",
    resolutions=["0.5m", "1.0m", "2.0m"],
    series_to_metrics={
        "ResNet-UNet": [metrics_05m, metrics_1m, metrics_2m],
        "SegFormer": [metrics_05m_sf, metrics_1m_sf, metrics_2m_sf]
    },
    metric_keys=["IoU"]
)
```

---

## Complete Training Example

Here's a complete example putting everything together:

```python
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T

from DroneClassification.data.MemoryMapDataset import MemmapDataset
from DroneClassification.models import DeepLab, JaccardLoss
from DroneClassification.training_utils import (
    TrainingSession,
    setup_device,
    calculate_class_weights,
    visualize_segmentation_results
)

# 1. Setup
device = setup_device()
classes = {0: "Background", 1: "Mangrove"}

# 2. Calculate class weights for imbalanced data
weights = calculate_class_weights("data/labels.npy", classes, power=2.0)

# 3. Create datasets with augmentation
train_transforms = T.Compose([
    T.RandomHorizontalFlip(0.5),
    T.RandomVerticalFlip(0.5),
    T.RandomRotation(15),
])

valid_transforms = T.Compose([])  # No augmentation for validation

train_dataset = MemmapDataset(
    images="data/train/images.npy",
    labels="data/train/labels.npy",
    transforms=train_transforms
)

valid_dataset = MemmapDataset(
    images="data/valid/images.npy",
    labels="data/valid/labels.npy",
    transforms=valid_transforms
)

# 4. Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True
)

# 5. Create model and loss
model = DeepLab(
    num_classes=1,
    input_image_size=224,
    backbone='resnet50'
)

loss_fn = JaccardLoss(
    num_classes=1,
    ignore_index=255,
    alpha=0.5,
    boundary_weight=0.1
)

# 6. Create and run training session
session = TrainingSession(
    model=model,
    trainLoader=train_loader,
    testLoader=valid_loader,
    lossFunc=loss_fn,
    num_epochs=100,
    init_lr=0.001,
    weight_decay=1e-4,
    experiment_name="deeplab_mangrove_v1",
    class_names=["Background", "Mangrove"],
    metric_mode="boundary"  # Include boundary IoU
)

session.learn()

# 7. Visualize some results
raw_images = np.load("data/valid/images.npy", mmap_mode='r')
for idx in [0, 10, 50]:
    visualize_segmentation_results(
        model=session.model,
        dataset=valid_dataset,
        raw_images=raw_images,
        sample_idx=idx,
        class_names=["Background", "Mangrove"],
        device=device
    )

# 8. Best model is automatically saved at:
# experiments/deeplab_mangrove_v1/best_model.pth
```

---

## Quick Reference

### TrainingSession Methods

| Method | Description |
|--------|-------------|
| `learn()` | Run the complete training loop |
| `evaluate(dataloader)` | Evaluate model on a dataset |
| `save_checkpoint(epoch, metrics)` | Manually save a checkpoint |
| `load_checkpoint(path)` | Resume from a checkpoint |
| `load_model_weights(path)` | Load only model weights |
| `save_model(path)` | Save only model weights |
| `plot_loss()` | Plot training/validation loss curves |
| `plot_metrics(title, metrics_wanted)` | Plot metric curves |
| `get_metrics()` | Get all validation metrics |
| `get_available_metrics()` | List available metric names |
| `register_metric_fn(name, fn)` | Add a custom metric function |

### Helper Functions

| Function | Description |
|----------|-------------|
| `setup_device()` | Auto-detect and return best available device |
| `calculate_class_weights(labels_path, classes)` | Compute weights for class imbalance |
| `visualize_segmentation_results(...)` | Create comprehensive visualization |
| `plot_comparison_metrics(...)` | Compare multiple models |
| `plot_loss_comparison(...)` | Compare loss curves from different runs |

---

## Next Steps

Now that you understand training, see:
- Inference Guide *(coming soon)* - How to deploy trained models on new imagery
