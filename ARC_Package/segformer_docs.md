# SegFormer Semantic Segmentation Toolbox for ArcGIS Pro

A professional, production-ready toolbox for running SegFormer semantic segmentation models directly in ArcGIS Pro.

## Features

✅ **Point-and-click interface** - No coding required  
✅ **Intelligent tile-based processing** - Handles rasters of any size  
✅ **GPU acceleration** - Automatic GPU detection and usage  
✅ **Batch processing** - Process multiple rasters automatically  
✅ **Progress tracking** - Real-time progress updates  
✅ **Model validation** - Verify models before processing  
✅ **Production-ready** - Error handling, logging, and optimization  

## Installation

### Step 1: Install Dependencies

Open the **Python Command Prompt** in ArcGIS Pro and run:

```bash
# Install PyTorch (CPU version)
conda install pytorch torchvision cpuonly -c pytorch

# Or for GPU support (CUDA 11.8)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Transformers library
pip install transformers

# Install GDAL if not already present
conda install gdal
```

### Step 2: Set Up Your SegFormer Module

1. Ensure your `SegFormer.py` file is in the correct location:
   ```
   C:\Users\gwrye\Documents\ArcGIS\Projects\MyProject1\SegFormer\SegFormer.py
   ```

2. If your SegFormer module is elsewhere, edit line 9 in `SegFormerToolbox.pyt`:
   ```python
   SEGFORMER_PATH = r"C:\Your\Custom\Path\SegFormer"
   ```

### Step 3: Add Toolbox to ArcGIS Pro

1. Save `SegFormerToolbox.pyt` to a convenient location (e.g., `C:\Users\gwrye\Documents\ArcGIS\Toolboxes\`)
2. Open **ArcGIS Pro**
3. In the **Catalog** pane, right-click **Toolboxes**
4. Select **Add Toolbox**
5. Browse to `SegFormerToolbox.pyt` and click **OK**

The toolbox will now appear in your Catalog with 4 tools!

## Tools Overview

### 1. Classify Raster with SegFormer

**Primary tool for semantic segmentation of single rasters.**

**Parameters:**
- **Input Raster**: The imagery to classify
- **Trained Model File**: Your trained .pth model file
- **Output Classified Raster**: Where to save results
- **Tile Size**: Size of processing tiles (default: 512px)
- **Tile Overlap**: Overlap between tiles for seamless blending (default: 64px)
- **Batch Size**: Number of tiles to process simultaneously (default: 4)
- **Use GPU**: Enable GPU acceleration if available
- **Class Names**: Comma-separated list of class names
- **Pretrained Weights**: Base model weights used during training
- **NoData Value**: Value to use for NoData pixels (default: 255)

**Example Usage:**
1. Open the tool from the Geoprocessing pane
2. Select your input imagery
3. Browse to your trained model (.pth file)
4. Specify output location
5. Adjust tile size based on your GPU memory (larger = faster but more memory)
6. Click **Run**

**Tips:**
- Start with default tile size (512) and adjust if needed
- Larger batch sizes are faster but require more GPU memory
- Use tile overlap to prevent edge artifacts
- GPU processing is 10-50x faster than CPU

### 2. Batch Classify Rasters

**Process multiple rasters automatically with the same model.**

**Parameters:**
- **Input Raster Folder**: Folder containing rasters to process
- **File Pattern**: Pattern to match files (e.g., `*.tif`, `ortho_*.tif`)
- **Trained Model File**: Your trained .pth model
- **Output Folder**: Where to save all results
- **Output Suffix**: Text to append to output filenames (default: `_classified`)
- **Tile Size**: Processing tile size
- **Class Names**: Comma-separated class names

**Example Usage:**
```
Input Folder: C:\Data\Imagery\
File Pattern: *.tif
Output Folder: C:\Data\Results\
Output Suffix: _segmented

Results:
- image1.tif → image1_segmented.tif
- image2.tif → image2_segmented.tif
- etc.
```

**Tips:**
- Process overnight for large batches
- Results are saved continuously (safe to stop and resume)
- Use consistent naming patterns for easier file management
- Check disk space before batch processing

### 3. Inspect Model Information

**View detailed information about your trained model.**

**Parameters:**
- **Model File**: The .pth model file to inspect

**Output Information:**
- File size
- Number of parameters
- Number of classes
- Layer architecture
- Checkpoint contents

**Example Usage:**
Use this tool to verify:
- Model loaded correctly
- Expected number of classes
- Model architecture details

### 4. Validate Model

**Test that a model can be loaded and run successfully.**

**Parameters:**
- **Model File**: The .pth model to validate
- **Number of Classes**: Expected number of output classes
- **Pretrained Weights**: Base model identifier

**Example Usage:**
Run this before processing to ensure:
- Model file is not corrupted
- Architecture matches training
- Model can perform inference
- No missing dependencies

## Workflow Guide

### Basic Workflow

```
1. Train your SegFormer model
   ↓
2. Save the trained model (.pth file)
   ↓
3. [Optional] Run "Validate Model" to verify it loads
   ↓
4. Run "Classify Raster with SegFormer"
   ↓
5. Review results and adjust parameters if needed
```

### Production Workflow

```
1. Validate model with test data
   ↓
2. Process a small test area first
   ↓
3. Verify accuracy and adjust tile size/overlap
   ↓
4. Use "Batch Classify Rasters" for full dataset
   ↓
5. Post-process results (filtering, smoothing, etc.)
```

## Troubleshooting

### "Out of Memory" Error

**Problem:** GPU runs out of memory during processing

**Solutions:**
- Reduce tile size (try 256 or 384)
- Reduce batch size (try 1 or 2)
- Close other GPU applications
- Use CPU instead (uncheck "Use GPU")

### "Module Not Found" Error

**Problem:** Cannot import SegFormer or dependencies

**Solutions:**
1. Verify SegFormer.py path in toolbox (line 9)
2. Install missing packages:
   ```bash
   pip install transformers torch
   ```
3. Check Python environment matches ArcGIS Pro's

### Slow Processing

**Problem:** Processing is very slow

**Solutions:**
- Enable GPU processing
- Increase batch size (if GPU memory allows)
- Increase tile size to 768 or 1024
- Reduce tile overlap to 32 pixels
- Check if other applications are using GPU

### Poor Edge Quality

**Problem:** Visible seams or artifacts at tile boundaries

**Solutions:**
- Increase tile overlap (try 128 or 256 pixels)
- Use smaller tile size for better blending
- Post-process with majority filter

### Wrong Number of Classes

**Problem:** Model outputs wrong number of classes

**Solutions:**
1. Run "Inspect Model Information" to check expected classes
2. Update "Class Names" parameter to match
3. Verify model was trained with correct number of classes

## Performance Optimization

### GPU Settings

| GPU Memory | Recommended Tile Size | Recommended Batch Size |
|------------|----------------------|------------------------|
| 4 GB       | 256                  | 2                      |
| 6 GB       | 384                  | 4                      |
| 8 GB       | 512                  | 4-8                    |
| 12 GB      | 512                  | 8-16                   |
| 16+ GB     | 768                  | 16-32                  |

### CPU Settings

For CPU processing:
- Tile Size: 512 (larger doesn't help much)
- Batch Size: 1-2 (CPU doesn't benefit from batching)
- Enable multi-threading in Windows (automatic)

### Large Raster Optimization

For very large rasters (>10,000 x 10,000 pixels):

1. Use tiled GeoTIFF format for output
2. Enable compression (automatically done)
3. Process in smaller geographic chunks if possible
4. Consider breaking into multiple smaller rasters

## Advanced Configuration

### Custom Class Colors

After processing, you can customize class colors:

1. Right-click the output raster → **Symbology**
2. Select **Unique Values**
3. Customize colors for each class
4. Save as layer file (.lyrx) for reuse

### Integration with Other Tools

Combine with ArcGIS tools:

```
SegFormer Classification
    ↓
Majority Filter (smoothing)
    ↓
Boundary Clean (edge refinement)
    ↓
Region Group (object identification)
    ↓
Raster to Polygon (vectorization)
```

### Automation with ModelBuilder

Create an automated workflow:

1. Open **ModelBuilder**
2. Add "Classify Raster with SegFormer" tool
3. Add post-processing tools
4. Save and run as batch process

### Python Scripting

Call the tools from Python scripts:

```python
import arcpy

# Set environment
arcpy.env.workspace = r"C:\Data"
arcpy.env.overwriteOutput = True

# Run classification
arcpy.segformer.ClassifyRasterWithSegFormer(
    input_raster=r"C:\Data\imagery.tif",
    model_file=r"C:\Models\segformer_trained.pth",
    output_raster=r"C:\Results\classified.tif",
    tile_size=512,
    tile_overlap=64,
    batch_size=4,
    use_gpu=True,
    class_names="Background,Building,Road,Vegetation",
    pretrained_weights="nvidia/segformer-b0-finetuned-ade-512-512",
    nodata_value=255
)

print("Classification complete!")
```

## Best Practices

### Model Training

1. **Consistent preprocessing**: Use same normalization during training and inference
2. **Sufficient training data**: More diverse data = better results
3. **Regular validation**: Test on held-out data during training
4. **Save checkpoints**: Keep multiple model versions

### Data Preparation

1. **Consistent resolution**: Match training data resolution
2. **Band alignment**: Ensure RGB bands in correct order
3. **Coordinate systems**: Match CRS between training and inference
4. **Data quality**: Remove clouds, shadows, artifacts before processing

### Production Processing

1. **Test first**: Always process a small area first
2. **Document settings**: Record tile size, overlap, and other parameters
3. **Quality control**: Manually review sample results
4. **Version control**: Track model versions and processing dates
5. **Backup originals**: Keep copies of input data

### Output Management

1. **Naming conventions**: Use descriptive, dated filenames
2. **Metadata**: Document processing parameters in file properties
3. **Compression**: Enable for storage efficiency (automatic)
4. **Pyramids**: Build for faster visualization (automatic)

## FAQ

**Q: Can I use models trained in other frameworks?**  
A: Currently supports SegFormer models trained with the provided SegFormer.py. Other architectures require code modifications.

**Q: How do I export results for use outside ArcGIS?**  
A: Outputs are standard GeoTIFF files compatible with QGIS, GDAL, Python, R, etc.

**Q: Can I classify multispectral imagery?**  
A: Yes, but the model uses only the first 3 bands (RGB). Update SegFormer.py for full multispectral support.

**Q: What's the maximum raster size?**  
A: No limit - tile-based processing handles any size raster within disk space constraints.

**Q: Can I run this on ArcGIS Server?**  
A: Not directly. This toolbox is for ArcGIS Pro desktop use only.

**Q: How accurate are the results?**  
A: Accuracy depends entirely on your training data quality and model training. The toolbox preserves model accuracy.

**Q: Can I use pre-trained models from Hugging Face?**  
A: Partially. You can use Hugging Face SegFormer weights as the base, but must fine-tune on your data first.

## Technical Details

### System Requirements

**Minimum:**
- ArcGIS Pro 2.9 or later
- 8 GB RAM
- 5 GB disk space
- Windows 10/11

**Recommended:**
- ArcGIS Pro 3.0+
- 16 GB RAM
- NVIDIA GPU with 6+ GB VRAM
- 50 GB disk space for large projects
- SSD for faster I/O

### Supported Data Formats

**Input:**
- GeoTIFF (.tif, .tiff)
- ERDAS IMAGINE (.img)
- Any GDAL-supported raster format

**Output:**
- Compressed GeoTIFF (LZW compression)
- 8-bit unsigned integer
- Single band (class labels)

### Processing Architecture

1. **Raster Reading**: Uses GDAL for efficient I/O
2. **Tiling**: Splits raster into overlapping tiles
3. **Batching**: Groups tiles for parallel GPU processing
4. **Inference**: PyTorch model on CPU or GPU
5. **Stitching**: Blends predictions from overlapping tiles
6. **Writing**: Streams results directly to disk

### Memory Management

- Tiles processed in batches to minimize memory
- Automatic cleanup after each batch
- Supports rasters larger than available RAM
- GPU memory automatically released

## Support & Resources

### Getting Help

1. Run "Validate Model" to diagnose model issues
2. Check ArcGIS Pro Python environment
3. Review error messages in Geoprocessing history
4. Check system resources (GPU memory, disk space)

### Additional Resources

- **SegFormer Paper**: https://arxiv.org/abs/2105.15203
- **ArcGIS Learn API**: https://developers.arcgis.com/python/api-reference/arcgis.learn.html
- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
- **Transformers Library**: https://huggingface.co/docs/transformers

### Updates & Maintenance

Keep your environment updated:

```bash
# Update PyTorch
conda update pytorch torchvision

# Update Transformers
pip install --upgrade transformers

# Update ArcGIS Pro
# Use ArcGIS Pro Package Manager
```

## Version History

**Version 1.0** (Current)
- Initial release
- Single raster classification
- Batch processing
- Model inspection tools
- GPU acceleration
- Tile-based processing with overlap blending

## License & Credits

Created for ArcGIS Pro integration of SegFormer semantic segmentation models.

**SegFormer**: Xie et al. (2021) - NVIDIA Research  
**Framework**: PyTorch, Hugging Face Transformers  
**Geospatial**: GDAL, ArcGIS Pro

---

**Need help?** Check the troubleshooting section or validate your model first!