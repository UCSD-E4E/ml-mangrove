import os
import fiona
import geopandas as gpd
import pandas as pd
from itertools import product
import rasterio
from rasterio.features import rasterize
from shapely.validation import make_valid
import numpy as np
from typing import Tuple

def tile_tiff_pair(chunk_path: str, image_size=128) -> Tuple[np.ndarray, np.ndarray]:
    name = chunk_path.split('/')[-1]
    print(f"Processing {name}...")

    
    name_list = name.split(' ')
    rgb_name = "Chunk" + name_list[1]
    if len(name_list) > 2:
        rgb_name += "_" + name_list[2]
    rgb_name += ".tif"

    rgb_path = os.path.join(chunk_path, rgb_name)
    label_path = os.path.join(chunk_path, "labels.tif")
    
    rgb_data, _ = read_tiff(rgb_path)
    label_data, label_meta = read_tiff(label_path)

    # Ensure we are only using the first three channels (RGB)
    if rgb_data.shape[0] > 3:
        rgb_data = rgb_data[:3, :, :]

    if label_meta['nodata'] is None:
        label_meta['nodata'] = 255  # Set nodata value if not defined

    # Ensure label_data has a single channel
    if label_data.shape[0] != 1:
        raise ValueError("Label TIFF should have a single channel.")

    # Create pairs of tiles
    images, labels = create_pairs(rgb_data, label_data, image_size)

    assert len(images) == len(labels), "Number of images and labels do not match."
    print(f"Number of valid pairs: {len(images)}")
    
    return images, labels

def rasterize_shapefile(chunk_path):
    name = chunk_path.split('/')[-1]
    
    # Get tiff path
    name_list = name.split(' ')
    tiff_name = "Chunk" + name_list[1]
    if len(name_list) > 2:
        tiff_name += "_" + name_list[2]
    tiff_name += ".tif"
    tiff_path = os.path.join(chunk_path, tiff_name)
    
    # Get labels
    shapefile_folder = os.path.join(chunk_path, 'labels')
    
    # Output label tif to Chunk folder
    output_path = os.path.join(chunk_path, 'labels.tif')
    if os.path.exists(output_path):
        print(f"{name} shapefile already rasterized")
        return
    else:
        print(f"Rasterizing {name} shapefile...")
    
    gdf = read_all_layers(shapefile_folder)
    if gdf is None:
        print(f"Failed to read {name} shapefile. Please check shape data.\n")
        return
    
    label_column = None
    for col in ['label', 'labels', 'class']:
        if col in gdf.columns:
            label_column = col
            break
    
    if not label_column:
        print(f"\n{name} shapefile does not have a 'label', 'labels', or 'class' column. Please check shape data.")
        print(f"Columns found: {gdf.columns}\n")
        return
    
    gdf[label_column] = gdf[label_column].replace({"1": 1, "0": 0, "mangrove": 1, "non-mangrove": 0})
    gdf[label_column] = pd.to_numeric(gdf[label_column], errors='coerce').replace([np.inf, -np.inf], np.nan).astype(np.float32)
    gdf[label_column] = gdf[label_column].apply(lambda x: np.nan if x > 255 else x).astype(np.float32)
    gdf[label_column] = gdf[label_column].fillna(255).astype(np.uint8)

    gdf = gdf[~gdf.geometry.isna()]  # Remove None geometries
    if not gdf.is_valid.all():
        print(f"\nInvalid geometries found in {name} shapefile, attempting to fix...")
        gdf['geometry'] = gdf['geometry'].buffer(0) # Fix simple invalid geometries
        if not gdf[~gdf.is_valid].empty:
            print("Using make_valid for more complex invalid geometry fix.")
            gdf.loc[~gdf.is_valid, 'geometry'] = gdf.loc[~gdf.is_valid, 'geometry'].apply(make_valid)
        if not gdf.is_valid.all():
            print(f"Failed to fix invalid geometries in {name} shapefile. Please check shape data.\n")
            return
        print(f"Fixed invalid geometries in {name} shapefile, but please validate results.\n")
    
    with rasterio.open(tiff_path) as src:
        meta = src.meta.copy()
        meta.update(dtype=rasterio.uint8, count=1, nodata=255)
        
        with rasterio.open(output_path, 'w', **meta) as out_raster:
            shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[label_column]))
            burned = rasterize(shapes=shapes, out_shape=src.shape, fill=255, transform=src.transform, dtype=rasterio.uint8)
            out_raster.write_band(1, burned)

def read_tiff(tif_path):
    with rasterio.open(tif_path) as src:
        data = src.read()
        meta = src.meta
    return data, meta

def tile_generator(data, tile_size):
    nrows, ncols = data.shape[1], data.shape[2]
    for i, j in product(range(0, nrows, tile_size), range(0, ncols, tile_size)):
        if i + tile_size <= nrows and j + tile_size <= ncols:
            yield data[:, i:i+tile_size, j:j+tile_size], (i, j)

def create_pairs(rgb_data, label_data, tile_size) -> Tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    for (rgb_tile, _), (label_tile, _) in zip(tile_generator(rgb_data, tile_size), tile_generator(label_data, tile_size)):
        if not np.any(label_tile == 255):  # Check for nodata values in label tile
            images.append(rgb_tile[:3, :, :])  # Keep only the first three channels (RGB)
            labels.append(label_tile)
    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Ensure images and labels have the same length
    assert images.shape[0] == labels.shape[0], "Mismatch in number of images and labels"
    
    return images, labels

def read_all_layers(shapefile_folder):
    shapefile_path = [os.path.join(shapefile_folder, f) for f in os.listdir(shapefile_folder) if f.endswith('.shp')][0]
    combined_gdf_list = []

    try:
        layers = fiona.listlayers(shapefile_path)
    except fiona.errors.DriverError as e:
        print(f"Error listing layers for {shapefile_path}: {e}")
        return None

    for layer in layers:
        try:
            gdf = gpd.read_file(shapefile_path, layer=layer)
        except fiona.errors.DriverError as e:
            print(f"Error reading layer {layer} in {shapefile_path}: {e}")
            continue
        
        if 'label' in gdf.columns:
            combined_gdf_list.append(gdf[['label', 'geometry']])
        else:
            combined_gdf_list.append(gdf)

    if combined_gdf_list:
        combined_gdf = pd.concat(combined_gdf_list, ignore_index=True)
        return combined_gdf
    else:
        print(f"No valid layers found in {shapefile_path}")
        return None