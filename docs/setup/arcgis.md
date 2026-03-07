# ArcGIS Pro Toolbox Setup

Deploy trained models as user-friendly tools in ArcGIS Pro for environmental scientists.

## How To Start

### 1. Install Dependencies

Install the latest version of **ArcGISPro**.
Install **Deep Learning Frameworks** for ArcGISPRO: found at [this repo.](https://github.com/Esri/deep-learning-frameworks?tab=readme-ov-file)


### 2. Verify Installation

Open **ArcGIS Pro → Python Command Prompt**, navigate to this directory, and run the following validation script:

```bash
cd ARC_Package
python install_validation.py
```

This script will give feedback on if there are still missing dependencies and if installation is sound. If the script gives the greenlight, the toolbox is good to use.

If extra dependencies need to be installed manually: consider the cloning the active environment @ ArcGIS Pro → Package Manager Tab → Gear Button In the Top Right.  This will set up a new, editable conda environment that packages can be installed to using the ArcGIS Pro's "Python Command Prompt" app that came with your installation.

Example commands for manual install (though, these may lead to conflicts):
```bash
# GPU version (recommended if you have NVIDIA GPU)
conda install pytorch torchvision pytorch-cuda=x.x -c pytorch -c nvidia
pip install transformers

# OR CPU version (no GPU)
conda install pytorch torchvision cpuonly -c pytorch
pip install transformers

# Verify GDAL is installed
conda list gdal
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
3. Double-click **Classify Raster**
4. Fill in parameters and run!













<!-- See [ArcGIS Pro Toolbox Setup](ARC_Package\README.md) for detailed instructions. -->
