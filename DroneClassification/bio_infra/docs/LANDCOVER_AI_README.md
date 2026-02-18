# Landcover.ai v1 Dataset

**Used for**: Testing Mamba-UNet architecture on aerial land cover segmentation

## Dataset Overview

- **Source**: [Landcover.ai v1](https://landcover.ai.linuxpolska.com/)
- **Purpose**: Automatic mapping of buildings, woodlands, water, and roads from aerial imagery
- **Location**: Poland, Central Europe
- **Coverage**: 216.27 km² total area
- **Resolution**:
  - 33 orthophotos at 25 cm/pixel resolution (~9000×9500 px)
  - 8 orthophotos at 50 cm/pixel resolution (~4200×4700 px)
- **Spectral Bands**: RGB (3 channels)
- **Format**: GeoTIFF with EPSG:2180 spatial reference

## Dataset Classes (Version 1)

| Class ID | Name | Area |
|----------|------|------|
| 0 | Background (non-mapped) | - |
| 1 | Building | 1.85 km² |
| 2 | Woodland | 72.02 km² |
| 3 | Water | 13.15 km² |
| 4 | Road | 3.5 km² |

## Directory Structure

```
landcover.ai.v1/
├── images/              # Full orthophoto GeoTIFFs (RGB, 1.5 GB total)
│   ├── M-33-20-D-c-4-2.tif
│   ├── M-33-20-D-d-3-3.tif
│   └── ... (33 + 8 = 41 GeoTIFFs)
│
├── masks/               # Segmentation masks GeoTIFFs (39 MB total)
│   ├── M-33-20-D-c-4-2.tif
│   ├── M-33-20-D-d-3-3.tif
│   └── ... (41 matching mask files)
│
├── split.py             # Python script to split large images into 512×512 tiles
├── train.txt            # List of training tile names
├── val.txt              # List of validation tile names
└── test.txt             # List of test tile names
```

## File Format Details

### Image Files
- **Filename format**: `{SHEET_ID}.tif` (e.g., `M-33-20-D-c-4-2.tif`)
- **Format**: 3-channel GeoTIFF (RGB)
- **Resolution**: Either 25 cm/pixel (~9000×9500 px) or 50 cm/pixel (~4200×4700 px)
- **Georeferencing**: EPSG:2180 coordinate system (Poland-specific)
- **Data type**: Uint8 (0-255 pixel values)

### Mask Files
- **Filename format**: Same as corresponding image file
- **Format**: Single-channel GeoTIFF
- **Resolution**: Matches corresponding image file
- **Georeferencing**: EPSG:2180 (same as images)
- **Data type**: Uint8
- **Values**:
  - 0 = Background (unmapped areas)
  - 1 = Buildings
  - 2 = Woodlands
  - 3 = Water
  - 4 = Roads

## Data Splits

The dataset is pre-split using tile-based cross-validation:

### Tile Definition
- **split.py** script converts the large orthophotos into 512×512 pixel tiles
- Each tile from each image gets a unique ID: `{image_name}_{tile_number}`
- Example: `M-33-20-D-c-4-2_0`, `M-33-20-D-c-4-2_1`, etc.

### Split Files
- **train.txt**: List of training tile IDs (majority)
- **val.txt**: List of validation tile IDs
- **test.txt**: List of test tile IDs

Each line contains a single tile identifier used to locate corresponding 512×512 crops within the large GeoTIFFs.
