# Models and Loss Functions Guide

This guide explains the neural network architectures and loss functions available for mangrove segmentation. Understanding which model and loss to choose and why can make a significant difference in your results.

---

## What is Semantic Segmentation?

Before diving into the models, it helps to understand the task they're solving. Semantic segmentation is about assigning a class label to every pixel in an image. For mangrove detection, this means looking at an aerial image and deciding, for each individual pixel, whether it shows mangrove or not.

This is fundamentally different from image classification (which assigns one label to the whole image) or object detection (which draws boxes around objects). Segmentation needs to understand both the overall context of the scene and the fine details of object boundaries.

The models in this repository are all designed for this task, but they take different approaches with different trade-offs in accuracy, speed, and memory usage.

---

## Where Everything Lives

All model code is in `DroneClassification/models/`:

- **models.py** - Neural network architectures (ResNet-UNet, DenseNet-UNet, SegFormer, DeepLab, etc.)
- **loss.py** - Loss functions for training (Jaccard, Dice, Focal, Boundary losses, etc.)
- **utils.py** - Helper functions used by the loss calculations

You can import everything directly:

```python
from DroneClassification.models import ResNet_UNet, DeepLab, SegFormer
from DroneClassification.models import JaccardLoss, DiceLoss, FocalLoss
```

---

## Understanding the Architecture Choices

### The Encoder-Decoder Pattern

Most segmentation models follow an encoder-decoder pattern. The encoder (also called the "backbone") takes the image and progressively shrinks it while extracting increasingly abstract features. Think of it as understanding "what" is in the image at the cost of losing spatial precision.

The decoder then takes those abstract features and progressively expands them back to the original image size, recovering the spatial detail needed to make per-pixel predictions.

The challenge is that the encoder loses spatial information as it shrinks the image. To compensate, most architectures use "skip connections", where it direct links from encoder layers to decoder layers, which help the decoder recover fine details like object edges.

### Transfer Learning

All models in this repository use pretrained encoders. This means the backbone was already trained on millions of images (usually ImageNet, a dataset of everyday objects). Even though ImageNet doesn't contain mangroves or aerial imagery, the low-level features it learned (edges, textures, shapes) transfer well to other domains.

By default, the encoder weights are frozen—they don't change during training. Only the decoder learns. This prevents the model from "forgetting" the useful features it already knows, and it means you can train with less data. If you have a lot of data, you can unfreeze the encoder and fine-tune everything.

---

## Available Models

### ResNet_UNet

The ResNet_UNet combines a ResNet18 encoder (a classic image classification network) with a UNet-style decoder. This is a good default choice, as it's fast, memory-efficient, and works well on most problems.

```python
from DroneClassification.models import ResNet_UNet

model = ResNet_UNet(
    input_image_size=224,  # Must match your tile size
    num_classes=1          # 1 for binary (mangrove/not), more for multi-class
)
```

**When to use it:**
- You want a reliable baseline that trains quickly
- You're working with limited GPU memory
- Your images are at a reasonable resolution where local context is enough

**How it works:**
The ResNet encoder has four main "layers" that progressively downsample the image (by factors of 4, 8, 16, 32). The decoder reverses this process, and skip connections from each encoder layer help the decoder recover spatial detail. The final output is a segmentation mask the same size as the input.

**Fine-tuning the backbone:**
By default, the ResNet weights are frozen. If you want to train the entire network:

```python
model.train_backbone(True)   # Enable gradient updates for encoder
model.train_backbone(False)  # Freeze encoder again
```

---

### DenseNet_UNet

Similar to ResNet_UNet, but uses a DenseNet121 backbone instead. DenseNet uses "dense connections" where each layer receives features from all previous layers. This creates very strong gradient flow and feature reuse.

```python
from DroneClassification.models import DenseNet_UNet

model = DenseNet_UNet(input_image_size=256)
```

**When to use it:**
- You want slightly better accuracy than ResNet_UNet
- You're okay with somewhat slower training
- DenseNet121 has fewer parameters (8M vs 11M) but is more compute-intensive due to the dense connections

**Trade-offs:**
DenseNet typically produces smoother segmentation boundaries. However, the dense connections mean more memory usage during training due to all the intermediate activations that need to be stored for backpropagation.

---

### SegFormer

SegFormer is a modern transformer-based architecture that has become very popular for segmentation tasks. Unlike CNNs (ResNet, DenseNet) that look at local regions through small convolutional filters, transformers use "attention" to look at relationships between all parts of the image at once.

```python
from DroneClassification.models import SegFormer

model = SegFormer(
    num_classes=1,
    input_image_size=512,
    weights="nvidia/segformer-b2-finetuned-ade-512-512"
)
```

**Available pretrained weights:**

The model name (b0 through b5) indicates the model size—larger models are more accurate but slower:

| Model | Parameters | Recommended Use |
|-------|------------|-----------------|
| b0 | ~4M | Fast inference, limited GPU |
| b1 | ~14M | Good balance |
| b2 | ~25M | Better accuracy (default) |
| b3 | ~45M | High accuracy |
| b4 | ~62M | Best accuracy, needs more GPU memory |
| b5 | ~82M | Maximum capacity |

You can choose from models pretrained on ADE20K (indoor/outdoor scenes) or Cityscapes (street scenes):

```python
# ADE20K pretrained (general scenes)
weights = "nvidia/segformer-b2-finetuned-ade-512-512"

# Cityscapes pretrained (street scenes - higher resolution)
weights = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
```

**When to use it:**
- You have larger images (512×512 or bigger)
- You need the model to understand long-range context (e.g., the relationship between a mangrove patch and the water nearby)
- You have a decent GPU (transformers are memory-hungry)

**Trade-offs:**
SegFormer typically achieves better accuracy than CNN-based models, especially on complex scenes. However, it requires more GPU memory and is slower to train. The attention mechanism also means inference time scales quadratically with image size.

---

### DeepLab

DeepLabV3 is another highly effective architecture that uses "atrous convolutions" (also called dilated convolutions). These are clever modifications to standard convolutions that let the model see a wider context without increasing the number of parameters or reducing resolution.

```python
from DroneClassification.models import DeepLab

model = DeepLab(
    num_classes=1,
    input_image_size=512,
    backbone='resnet50',    # or 'resnet101', 'mobilenet_v3_large'
    output_stride=8         # Controls spatial resolution in the encoder
)
```

**Backbone options:**
- `resnet50` - Good balance of speed and accuracy (default)
- `resnet101` - More capacity, better for complex scenes
- `mobilenet_v3_large` - Much faster, good for edge deployment
- `xception` - Uses depthwise separable convolutions
- `segformer` - Combines DeepLab's ASPP with SegFormer's encoder

**What is ASPP?**
The key innovation in DeepLab is the Atrous Spatial Pyramid Pooling (ASPP) module. It applies multiple parallel convolutions at different dilation rates, allowing the model to capture objects at multiple scales simultaneously. This is particularly useful for aerial imagery where mangrove patches can vary dramatically in size.

**Output stride:**
The `output_stride` parameter controls how much the spatial resolution is reduced in the encoder:
- `output_stride=16` - Standard, faster but lower resolution features
- `output_stride=8` - Higher resolution features, better boundary accuracy
- `output_stride=4` - Highest resolution, most memory intensive

Lower output strides preserve more spatial detail but use more memory.

**When to use it:**
- You're dealing with objects at multiple scales
- You need precise boundaries
- You want a well-established architecture with lots of research backing

---

### Xception

Xception ("Extreme Inception") uses depthwise separable convolutions—a technique that factorizes standard convolutions into two steps, dramatically reducing computation while maintaining accuracy.

```python
from DroneClassification.models import Xception

model = Xception(num_classes=1)
```

**When to use it:**
- You need efficient computation
- You're targeting mobile or edge deployment
- You want a good accuracy-to-compute ratio

The Xception backbone in this repo is primarily used as a backbone option for DeepLab, but can be used standalone.

---

### ResNet_FC (Fully Connected)

This is a simpler architecture that uses ResNet as a feature extractor and a fully connected layer to produce the segmentation. It's included for experimentation but generally not recommended for production use.

```python
from DroneClassification.models import ResNet_FC

model = ResNet_FC(num_classes=1, input_image_size=128)
```

**When to use it:**
- Experimentation only
- Very small images
- When you want to understand baseline performance

**Limitations:**
The fully connected layer means the model is tied to a specific input size and loses all spatial structure. UNet-style decoders generally perform much better.

---

## Making Predictions with Trained Models

Once you've trained a model, you'll want to use it for inference. The `SegmentModelWrapper` class simplifies this by handling normalization and thresholding:

```python
from DroneClassification.models.models import SegmentModelWrapper

# Wrap your trained model
wrapper = SegmentModelWrapper(model, threshold=0.5)

# Now you can pass raw images (0-255) without manual preprocessing
prediction = wrapper(image)  # Returns binary mask (0 or 1)
```

The wrapper:
- Accepts numpy arrays or PyTorch tensors
- Automatically scales images from 0-255 to 0-1
- Applies ImageNet normalization
- Applies sigmoid and thresholding to produce binary output

---

## Loss Functions

Loss functions measure how wrong the model's predictions are. During training, the optimizer adjusts the model weights to minimize this loss. Different loss functions emphasize different aspects of the problem.

### The Class Imbalance Problem

In mangrove segmentation, you often have far more "not mangrove" pixels than "mangrove" pixels. If 95% of pixels are background, a model that predicts "not mangrove" for everything would have 95% accuracy—but be completely useless.

Most loss functions in this repo are designed to handle this imbalance, either through weighting or by using metrics that don't reward trivial predictions.

---

### JaccardLoss (IoU Loss)

Jaccard Loss directly optimizes the Intersection over Union (IoU) metric, which is often what you actually care about in segmentation tasks.

```python
from DroneClassification.models import JaccardLoss

criterion = JaccardLoss(
    num_classes=1,         # Binary segmentation
    ignore_index=255,      # Pixels with this label are ignored
    alpha=0.5,             # Balance between CE and Jaccard (0.5 = equal)
    boundary_weight=0.0    # Optional: add boundary loss component
)

loss = criterion(logits, labels)
```

**How it works:**
IoU measures the overlap between prediction and ground truth:

```
IoU = Intersection / Union
    = (Predicted AND Correct) / (Predicted OR Correct)
```

A perfect prediction has IoU = 1. The Jaccard Loss is simply `1 - IoU`.

**The alpha parameter:**
Pure Jaccard Loss can have gradient issues early in training when predictions are poor. The `alpha` parameter blends it with Cross-Entropy (or BCE for binary):

```
Loss = alpha * CrossEntropy + (1 - alpha) * JaccardLoss
```

Higher alpha puts more weight on CrossEntropy, which has better gradients but doesn't directly optimize IoU.

**Boundary loss component:**
Setting `boundary_weight > 0` adds Active Boundary Loss (explained below), which helps sharpen object edges.

**When to use it:**
- Your evaluation metric is IoU/mIoU
- You have class imbalance
- Good default choice for segmentation

---

### DiceLoss

Dice Loss is closely related to Jaccard but uses the Dice coefficient (also called F1 score):

```python
from DroneClassification.models import DiceLoss

criterion = DiceLoss(
    num_classes=1,
    ignore_index=255
)
```

**How it works:**

```
Dice = 2 * Intersection / (Prediction + GroundTruth)
```

Dice and Jaccard are mathematically related: `Dice = 2*IoU / (1+IoU)`. In practice, they often produce similar results, but Dice Loss tends to have slightly better gradients.

**When to use it:**
- Similar situations to Jaccard
- Some practitioners prefer it for medical imaging
- If Jaccard isn't converging well, try Dice

---

### DiceJaccardLoss

A combination that uses both Dice and Jaccard:

```python
from DroneClassification.models.loss import DiceJaccardLoss

criterion = DiceJaccardLoss(num_classes=1)
# Internally uses: 0.3 * DiceLoss + 0.7 * JaccardLoss
```

This can provide a good balance of the gradient properties of both losses.

---

### FocalLoss

Focal Loss was designed for object detection but works well for segmentation with severe class imbalance. It down-weights easy examples (pixels the model is already confident about) and focuses training on hard examples.

```python
from DroneClassification.models import FocalLoss

criterion = FocalLoss(
    alpha=0.1,          # Weight for positive class
    gamma=2,            # Focus parameter (higher = more focus on hard examples)
    ignore_index=255
)
```

**How it works:**
Standard Cross-Entropy treats all examples equally. Focal Loss adds a modulating factor:

```
FL = -(1 - p)^gamma * log(p)
```

When `gamma = 0`, this is just Cross-Entropy. When `gamma > 0`, easy examples (where p is high) are down-weighted. The effect is that the model spends more effort on the pixels it's uncertain about.

**The gamma parameter:**
- `gamma = 0`: Standard Cross-Entropy
- `gamma = 1`: Mild focusing
- `gamma = 2`: Standard focal loss (default)
- `gamma = 5`: Strong focusing on hard examples

**When to use it:**
- Severe class imbalance
- Model keeps predicting all background
- You want the model to focus on difficult boundary regions

---

### WeightedMultiClassFocalLoss

An extension of Focal Loss for multi-class problems with per-class weights:

```python
from DroneClassification.models.loss import WeightedMultiClassFocalLoss

criterion = WeightedMultiClassFocalLoss(
    alpha=[0.1, 0.3, 0.6],  # Per-class weights
    gamma=2.0,
    ignore_index=255
)
```

Use this when you have multiple classes (not just mangrove/not-mangrove) with different frequencies.

---

### FocalTverskyLoss

Focal Tversky Loss lets you explicitly control the trade-off between false positives and false negatives:

```python
from DroneClassification.models.loss import FocalTverskyLoss

criterion = FocalTverskyLoss(
    alpha=0.7,   # Weight for false negatives (higher = punish FN more)
    beta=0.3,    # Weight for false positives (should be 1 - alpha)
    gamma=0.75   # Focal parameter
)
```

**Understanding alpha and beta:**
- High `alpha` (e.g., 0.7): The model is penalized more for missing mangroves (false negatives)
- High `beta` (e.g., 0.7): The model is penalized more for false alarms (false positives)

**When to use it:**
- You care more about recall (finding all mangroves) than precision
- Or vice versa—you need to minimize false alarms
- When the cost of different errors is asymmetric

---

### ActiveBoundaryLoss

This specialized loss helps the model produce sharper, more accurate boundaries. It works by computing the KL divergence between predictions at boundary pixels and their neighbors, then using distance transforms to weight the contribution of each pixel.

```python
from DroneClassification.models.loss import ActiveBoundaryLoss

boundary_loss = ActiveBoundaryLoss(
    num_classes=1,
    ignore_index=255,
    max_N_ratio=1/100,    # Max fraction of pixels to consider as boundary
    max_clip_dist=20.0    # Max distance for weighting
)
```

**When to use it:**
- You need precise boundaries (e.g., for area calculations)
- Your predictions have fuzzy or jagged edges
- Best used in combination with another loss (like JaccardLoss with `boundary_weight > 0`)

**Note:** Boundary loss is computationally expensive. Use it when boundary precision really matters, not as a default.

---

### LandmassLoss

A simple auxiliary loss that encourages the model to predict the correct total area of mangroves:

```python
from DroneClassification.models.loss import LandmassLoss

criterion = LandmassLoss()
loss = criterion(prediction, target)  # Normalized difference in total area
```

**When to use it:**
- You care about accurate area estimates
- Use as an auxiliary loss alongside Jaccard/Dice, not alone

---

## Choosing the Right Model and Loss

### Quick Recommendations

**For getting started:**
- Model: `ResNet_UNet` or `DeepLab(backbone='resnet50')`
- Loss: `JaccardLoss(alpha=0.5)`

**For best accuracy (if you have a good GPU):**
- Model: `SegFormer(weights="nvidia/segformer-b3-finetuned-ade-512-512")`
- Loss: `JaccardLoss(alpha=0.3, boundary_weight=0.1)`

**For fast inference:**
- Model: `DeepLab(backbone='mobilenet_v3_large')`
- Loss: `JaccardLoss(alpha=0.5)`

**For severe class imbalance:**
- Model: Any
- Loss: `FocalLoss(gamma=2)` or `FocalTverskyLoss(alpha=0.7)`

**For precise boundaries:**
- Model: `DeepLab(output_stride=4)` or `SegFormer`
- Loss: `JaccardLoss(boundary_weight=0.2)`

---

## Quick Reference

### Models

| Model | Backbone | Best For | Memory Usage |
|-------|----------|----------|--------------|
| `ResNet_UNet` | ResNet18 | General use, fast training | Low |
| `DenseNet_UNet` | DenseNet121 | Smooth boundaries | Medium |
| `SegFormer` | MiT-B0 to B5 | Best accuracy, large images | High |
| `DeepLab` | ResNet50/101, MobileNet | Multi-scale objects | Medium-High |
| `Xception` | Xception | Efficient computation | Medium |

### Loss Functions

| Loss | Handles Imbalance | Optimizes For | Use Case |
|------|-------------------|---------------|----------|
| `JaccardLoss` | Yes | IoU | Default choice |
| `DiceLoss` | Yes | Dice/F1 | Similar to Jaccard |
| `FocalLoss` | Yes (strongly) | Hard examples | Severe imbalance |
| `FocalTverskyLoss` | Yes | FP/FN trade-off | Asymmetric errors |
| `ActiveBoundaryLoss` | No | Sharp boundaries | Boundary precision |

---

## Next Steps

Now that you understand the models and losses, see:
- Training Guide *(coming soon)* - How to train these models
- Inference Guide *(coming soon)* - How to deploy trained models
