# Semantic Segmentation Toolbox

**A professional, production-ready toolbox for ArcGIS Pro**

Transform your trained models into user-friendly point-and-click tools that integrate seamlessly with ArcGIS Pro's Geoprocessing framework.

## 🎯 Quick Start

### 1. Install Dependencies

Open **ArcGIS Pro → Python Command Prompt** and run:

```bash
# GPU version (recommended if you have NVIDIA GPU)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers

# OR CPU version (no GPU)
conda install pytorch torchvision cpuonly -c pytorch
pip install transformers

# Verify GDAL is installed
conda list gdal
```

### 2. Verify Installation

Run the validation script:

```bash
python install_validation.py
```

### 3. Add Toolbox to ArcGIS Pro

1. Save `SegmentationToolbox.pyt` to your preferred location
2. In ArcGIS Pro, open **Catalog** pane
3. Right-click **Toolboxes** → **Add Toolbox**
4. Browse to `SegmentationToolbox.pyt`
5. Click **OK**

### 4. Start Classifying!

1. Open **Geoprocessing** pane
2. Navigate to **SegFormer Semantic Segmentation** toolbox
3. Double-click **Classify Raster with SegFormer**
4. Fill in parameters and run!

## 📦 Package Contents

```
Segmentation-Toolbox/
│
├── SegmentationToolbox.pyt        # Main ArcGIS Pro toolbox
├── install_validation.py          # Installation verification script
├── README.md                      # This file
├── USER_GUIDE.md                  # Comprehensive user documentation
│
│
├──SegFormer.py/                     # Your model definition module
│
└── SegFormer.pth/                   # Your pretrained model weights
```

## 🛠️ Tools Included

### 1️⃣ Classify Raster with SegFormer
Main tool for semantic segmentation of imagery
- Smart tile-based processing for any raster size
- GPU acceleration for speedup
- Automatic overlap blending for seamless results

### 3️⃣ Inspect Model Information
View model details before processing
- Number of classes
- Parameter count
- Architecture details

### 4️⃣ Validate Model
Test model loading and inference
- Verify model integrity
- Check compatibility
- Test inference capability

### Customize Default Parameters

Edit the `getParameterInfo()` methods in each tool to change defaults:

```python
params[3].value = 512  # Default tile size
params[4].value = 64   # Default overlap
params[5].value = 4    # Default batch size
```

## 🐛 Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'transformers'`

**Solution:**
```bash
pip install transformers
```

### GPU Not Detected

**Problem:** Tool says "Using CPU" but you have a GPU

**Solution:**
1. Install GPU version of PyTorch
2. Update NVIDIA drivers
3. Verify CUDA compatibility

### Out of Memory

**Problem:** `CUDA out of memory` error

**Solution:**
- Reduce tile size (try 256)
- Reduce batch size (try 1 or 2)
- Close other GPU applications
- Use CPU mode

### Tile Seams Visible

**Problem:** Visible edges between tiles in output

**Solution:**
- Increase tile overlap (try 128 or 256)
- Apply majority filter in post-processing

## 📚 Documentation

- **Tool help** - Click (?) in ArcGIS Pro tool dialog
- **SegFormer paper** - https://arxiv.org/abs/2105.15203

## 🔄 Updates

### Version 1.0 (Current)
- ✅ raster classification
- ✅ Batch processing
- ✅ Model validation tools
- ✅ GPU acceleration
- ✅ Tile-based processing
- ✅ Progress tracking
- ✅ Error handling