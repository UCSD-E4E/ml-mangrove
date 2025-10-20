# Human Infrastructure Normalization Pipeline

## Overview
Basic framework for normalizing different data sources in the Human Infrastructure project.

## Features
- Abstract base class for normalization strategies
- ImageNet normalization for RGB images
- Min-max normalization for other data types
- Pipeline for handling multiple data sources
- Basic test suite

## Usage

```python
from normalization_pipeline import create_default_pipeline, normalize_for_training

# Quick normalization for training
normalized_data = normalize_for_training(rgb_image)

# Custom pipeline
pipeline = create_default_pipeline()
normalized = pipeline.normalize_data("rgb_images", data)
```

## Files Added
- `normalization_pipeline.py` - Main pipeline implementation
- `test_normalization.py` - Basic test suite
