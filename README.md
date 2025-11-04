# Mangrove Monitoring: Machine Learning

This repo includes all the tools and testing related to ML development for the Mangrove Monitoring Project, which consists of four main areas: Aerial Mangrove Identification, Drone-Satellite Fusion Methods, Human Infrastructure Identification, and an ArcGIS Toolbox package.

## Mangrove ID
Binary segmentation on aerial images to label pixels as mangrove or not mangrove.

## Human Infrastructure ID
Detecting human-made structures (roads, buildings) that threaten mangroves. Currently seeking appropriate datasets that generalize well to mangrove environments.

**Todo:**
- Source labeled human infrastructure vs natural features data
- Implement overlap in tile processing
- Consider creating labeled human infrastructure data from our mangrove imagery
- Normalize and combine multiple data sources

## Satellite Super-Res
Aimed to enhance satellite imagery for global mangrove tracking without drones. Discontinued because classifiers don't function well even at super-resolved resolution, but it is a useful tool to have.

**Todo:**
- Implement image space Schrödinger Bridge Diffusion
- Edit Schrödinger Bridge Latent Diffusion model to extract features using standard resnet. The multispectral resnet feature space is incompatible with the target rgb feature space, so we need to use the same standard resnet feature extractor for both. The only issue is we need to find a way to incorporate the multispectral channels into the process.
- Find pretrained super-res models
- Gather a dataset

## ArcGIS Toolbox
Drop-in toolbox for ArcGIS Pro enabling environmental scientists to easily deploy and use our models for imagery classification.

**Todo:**
- Filter models by task in UI
- Package model info into .emd files
- Trim edge predictions with no input
- Add overlap blending logic
- Implement batch processing for GPU acceleration

## Our Pipeline
![pipeline](readme_resources/Mangrove_Pipeline.jpeg)

### 1. Data Processing
See `DroneClassification/data/process_data` notebook for processing geospatial imagery. All tools are stored in the `utils.py` file.

### 2. Model Training
Model architectures and loss functions are in `DroneClassification/models`. See `model_training_ground` notebook for training template.

**Current Models:**
- ResNet18 UNet: Best for Mangrove Classification
- SegFormer B0/B2: Best for Human Infrastructure 

**Super-Resolution (testing):**
- Schrödinger Bridge Latent Diffusion
- 3-layer SRCNN

### 3. ArcGIS Packaging
`ARC_Package` contains the toolbox and model formatting template. Each architecture requires a ModelClass to be implemented.
