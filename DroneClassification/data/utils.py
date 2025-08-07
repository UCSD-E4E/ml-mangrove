import os
import gc
import fiona
import numpy as np
import geopandas as gpd
import pandas as pd
import shutil
import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from itertools import product
from shapely.validation import make_valid
from shapely.geometry import shape, Polygon, MultiPolygon
from typing import Tuple
from pyproj import CRS
from tqdm import tqdm


def tile_dataset(data_path: str, combined_images_file: str, combined_labels_file: str, chunk_buffer_size: int=1, image_size=224):
        
    # Buffer for storing data before appending to memmap
    image_buffer = []
    label_buffer = []

    num_chunks = len([entry for entry in os.listdir(data_path) if 'Chunk' in entry])
    print(f"Processing {num_chunks} chunk directories")

    # Iterate over each chunk directory and process TIFF pairs
    current_chunk = 0
    for entry in os.listdir(data_path):
        if 'Chunk' in entry:
            current_chunk += 1
            chunk_name = os.path.basename(entry)

            print(f"\n[{current_chunk}/{num_chunks}] Processing chunk: {chunk_name}...")
            chunk_path = os.path.join(data_path, entry)
            
            # Generate tiled images and labels
            images, labels = _tile_tiff_pair(chunk_path, image_size=image_size)
            if images.size == 0:
                print(f"No valid tiles found at {chunk_name}")
                continue
            print(f"Number of valid tiles found: {len(images)}")
            
            # Add to buffer
            image_buffer.append(images)
            label_buffer.append(labels)

            if current_chunk % chunk_buffer_size == 0:

                print(f"Buffer size reached {chunk_buffer_size} chunks, appending to memmap.")

                images_to_append = np.concatenate(image_buffer, axis=0)
                _append_to_memmap(combined_images_file, images_to_append, np.uint8)
                image_buffer = []
                del images_to_append

                labels_to_append = np.concatenate(label_buffer, axis=0)
                _append_to_memmap(combined_labels_file, labels_to_append, np.uint8)
                label_buffer = []
                del labels_to_append
                gc.collect()

    # Final append if buffer is not empty
    if image_buffer:
        print("Appending remaining buffered data to memmap.")

        images_to_append = np.concatenate(image_buffer, axis=0)
        _append_to_memmap(combined_images_file, images_to_append, np.uint8)
        image_buffer = []
        del images_to_append

        labels_to_append = np.concatenate(label_buffer, axis=0)
        _append_to_memmap(combined_labels_file, labels_to_append, np.uint8)
        label_buffer = []
        del labels_to_append

    print('\nDone tiling tif pairs')

def rasterize_shapefiles(path):
    """
    Rasterizes the shapefiles in the given path.
    Args:
        path (str): Path to the directory of chunks containing the shapefile and TIFF.
    """

    for entry in tqdm(os.listdir(path), desc="Searching for shapefiles", unit="file"):
        if 'Chunk' in entry:
            chunk_path = os.path.join(path, entry)
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
                return
            else:
                print(f"Rasterizing {name} shapefile...")
            
            gdf = _read_all_layers(shapefile_folder)
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
                print(f"Fixed invalid geometries in {name} shapefile.\n")
            
            with rasterio.open(tiff_path) as src:
                meta = src.meta.copy()
                meta.update(dtype=rasterio.uint8, count=1, nodata=255)
                
                with rasterio.open(output_path, 'w', **meta) as out_raster:
                    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[label_column]))
                    burned = rasterize(shapes=shapes, out_shape=src.shape, fill=255, transform=src.transform, dtype=rasterio.uint8)
                    out_raster.write_band(1, burned)
        print('\nDone rasterizing shapefiles')

def resample(input_root: str, output_root: str, target_resolution: float, save_backup_labels: bool = False):
    """
    
    Resamples TIFF files to a specified resolution and rasterizes/aligns shapefiles to the resampled TIFFs.

    It expects a directory where each TIFF is in its own folder, along with its label files in a subfolder named "labels".

    Args:
        input_root (str): Path to the directory containing original TIFF files.
        output_root (str): Path to the directory where resampled TIFFs will be saved.
        target_resolution (float): Target resolution in meters.
        save_backup_labels (bool): If True, original labels will be moved to a backup directory.
    """


    if not os.path.isdir(input_root):
        raise NotADirectoryError(f"{input_root} is not a valid directory.")
    
    # Resample TIFFs
    for root, _, files in tqdm(os.walk(input_root), desc="Resampling TIFFs", unit="folder"):
        for file in files:
            if file.endswith(".tif"):
                tiff_path = os.path.join(root, file)
                subfolder = os.path.relpath(root, input_root)
                output_dir = os.path.join(output_root, subfolder)
                os.makedirs(output_dir, exist_ok=True)
                output_tiff_path = os.path.join(output_dir, f"{file}")

                with rasterio.open(tiff_path) as src:
                    # Get center of the raster
                    lon = (src.bounds.left + src.bounds.right) / 2
                    lat = (src.bounds.top + src.bounds.bottom) / 2
                    dst_crs = _get_utm_crs(lon, lat)

                    transform, width, height = calculate_default_transform(
                        src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=target_resolution
                    )

                    kwargs = src.meta.copy()
                    kwargs.update({
                        'crs': dst_crs,
                        'transform': transform,
                        'width': width,
                        'height': height
                    })

                    resampling = Resampling.bilinear

                    with rasterio.open(output_tiff_path, 'w', **kwargs) as dst:
                        for i in range(1, src.count + 1):
                            reproject(
                                source=rasterio.band(src, i),
                                destination=rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=dst_crs,
                                resampling=resampling
                            )
                # Copy labels if present
                labels_dir = os.path.join(root, "labels")
                if os.path.exists(labels_dir):
                    output_labels_dir = os.path.join(output_dir, "labels")
                    os.makedirs(output_labels_dir, exist_ok=True)
                    for label_file in os.listdir(labels_dir):
                        src_file = os.path.join(labels_dir, label_file)
                        dst_file = os.path.join(output_labels_dir, label_file)
                        shutil.copy(src_file, dst_file)

    print("Aligning shapefiles to resampled TIFFs...")
    _align_shapefiles(output_root, save_backup_labels=save_backup_labels)
    print(f"Resampling completed. Resampled TIFFs saved to {output_root}.")

def _align_shapefiles(output_root,
    simplify_tolerance=1.0,
    buffer_amount=0.5,
    min_area=10.0,
    save_backup_labels=False
):
    # Align shapefiles to resampled TIFFs
    for root, dirs, files in os.walk(output_root):
        has_labels=False
        for file in files:
            if file.endswith(".tif"):
                
                tiff_path = os.path.join(root, file)
                labels_dir = os.path.join(root, "labels")
                if os.path.isdir(labels_dir):
                    shapefiles = [f for f in os.listdir(labels_dir) if f.endswith(".shp")]
                    if not shapefiles:
                        continue  # no shapefile in labels/
                    has_labels=True
                    shapefile_path = os.path.join(labels_dir, shapefiles[0])
                    output_shapefile_path = os.path.join(labels_dir, f"{file[:-4]}_aligned.shp")
                    print(f"Rasterizing & smoothing shape file for {file}")
    
                    # Load TIFF to get size, transform, CRS
                    with rasterio.open(tiff_path) as src:
                        out_shape = (src.height, src.width)
                        transform = src.transform
                        crs = src.crs

                    # Load shapefile and reproject if needed
                    gdf = gpd.read_file(shapefile_path)

                    if gdf.crs is None:
                        print(f"⚠️  No CRS found in {shapefile_path}. Assigning from raster: {crs}")
                        gdf.set_crs(crs, inplace=True)  # Assume it already matches raster
                    elif gdf.crs != crs:
                        gdf = gdf.to_crs(crs)

                    # Rasterize shapefile
                    mask = rasterize(
                        [(geom, 1) for geom in gdf.geometry],
                        out_shape=out_shape,
                        transform=transform,
                        fill=0,
                        all_touched=True,
                        dtype='uint8'
                    )

                    # Polygonize the binary mask
                    results = (
                        (shape(geom), value)
                        for geom, value in shapes(mask, mask=(mask == 1), transform=transform)
                    )

                    # Smooth and clean geometries
                    cleaned = []
                    for geom, val in results:
                        smoothed = geom.buffer(buffer_amount).simplify(simplify_tolerance).buffer(-buffer_amount)

                        if isinstance(smoothed, Polygon):
                            if smoothed.area >= min_area:
                                cleaned.append({"geometry": smoothed, "value": val})
                        elif isinstance(smoothed, MultiPolygon):
                            for part in smoothed.geoms:
                                if part.area >= min_area:
                                    cleaned.append({"geometry": part, "value": val})

                    if not cleaned:
                        print("No valid shapes found after smoothing.")
                        continue

                    # Save result
                    out_gdf = gpd.GeoDataFrame(cleaned, crs=crs)
                    out_gdf.to_file(output_shapefile_path)
        
        # Move original labels to backup directory
        if has_labels and save_backup_labels: 
            labels_dir = os.path.join(root, "labels")
            backup_dir = os.path.join(root, "original_backup_labels")
            os.makedirs(backup_dir, exist_ok=True)
            first_file = True

            for fname in os.listdir(labels_dir):
                fpath = os.path.join(labels_dir, fname)

                if os.path.isfile(fpath) and "_aligned" not in fname and fname.endswith((".shp", ".shx", ".dbf", ".prj", ".cpg", ".qmd")):
                    if first_file:
                        print(f"⚠️  Moving {os.path.splitext(fname)[0]} to {backup_dir}")
                        first_file = False
                    shutil.move(fpath, os.path.join(backup_dir, fname))

def _append_to_memmap(file_path, data, dtype):
    if not os.path.exists(file_path):
        print(f"Creating new memmap file at {file_path}")
        new_memmap = np.lib.format.open_memmap(file_path, mode='w+', dtype=dtype, shape=data.shape)
        new_memmap[:] = data
    else:
        existing_shape = np.load(file_path).shape
        new_shape = (existing_shape[0] + data.shape[0],) + existing_shape[1:]

        temp_file_path = file_path + '.tmp'
        new_memmap = np.lib.format.open_memmap(temp_file_path, mode='w+', dtype=dtype, shape=new_shape)

        
        old_memmap = np.lib.format.open_memmap(file_path, mode='r')
        new_memmap[:existing_shape[0]] = old_memmap[:]
        new_memmap[existing_shape[0]:] = data

        new_memmap.flush()
        del new_memmap
        del old_memmap
        gc.collect()
        
        # Replace the original file with the temporary file
        os.replace(temp_file_path, file_path)

def _create_pairs(rgb_data, label_data, tile_size) -> Tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    for (rgb_tile, _), (label_tile, _) in zip(_tile_generator(rgb_data, tile_size), _tile_generator(label_data, tile_size)):
        if not np.any(label_tile == 255):  # Check for nodata values in label tile
            images.append(rgb_tile[:3, :, :])  # Keep only the first three channels (RGB)
            labels.append(label_tile)
    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Ensure images and labels have the same length
    assert images.shape[0] == labels.shape[0], "Mismatch in number of images and labels"
    
    return images, labels

def _get_utm_crs(lon, lat):
    """
    Get UTM CRS based on longitude and latitude.
    Args:
        lon (float): Longitude of the center point.
        lat (float): Latitude of the center point.
    Returns:
        CRS: UTM CRS for the given coordinates.
    """
    utm_zone = int((lon + 180) / 6) + 1
    is_northern = lat >= 0
    return CRS.from_dict({
        "proj": "utm",
        "zone": utm_zone,
        "datum": "WGS84",
        "south": not is_northern
    })
def _read_all_layers(shapefile_folder):
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

def _read_tiff(tif_path):
    with rasterio.open(tif_path) as src:
        data = src.read()
        meta = src.meta
    return data, meta

def _tile_generator(data, tile_size):
    nrows, ncols = data.shape[1], data.shape[2]
    for i, j in product(range(0, nrows, tile_size), range(0, ncols, tile_size)):
        if i + tile_size <= nrows and j + tile_size <= ncols:
            yield data[:, i:i+tile_size, j:j+tile_size], (i, j)

def _tile_tiff_pair(chunk_path: str, image_size=224) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a chunk directory containing a TIFF image and its corresponding label TIFF, creates tile pairs of each in the specified size,
    and returns the image and label tiles as numpy arrays.
    
    Args:
        chunk_path (str): Path to the chunk directory containing the TIFF and label TIFF.
        image_size (int): Size of the image tiles to be created.
    Returns:
        Tiles (Tuple): A tuple containing the image data and label data as numpy arrays.
    """
    name = chunk_path.split('/')[-1]
    print(f"Processing {name}...")

    
    name_list = name.split(' ')
    rgb_name = "Chunk" + name_list[1]
    if len(name_list) > 2:
        rgb_name += "_" + name_list[2]
    rgb_name += ".tif"

    rgb_path = os.path.join(chunk_path, rgb_name)
    label_path = os.path.join(chunk_path, "labels.tif")
    
    rgb_data, _ = _read_tiff(rgb_path)
    label_data, label_meta = _read_tiff(label_path)

    # Ensure we are only using the first three channels (RGB)
    if rgb_data.shape[0] > 3:
        rgb_data = rgb_data[:3, :, :]

    if label_meta['nodata'] is None:
        label_meta['nodata'] = 255  # Set nodata value if not defined

    # Ensure label_data has a single channel
    if label_data.shape[0] != 1:
        raise ValueError("Label TIFF should have a single channel.")

    # Create pairs of tiles
    images, labels = _create_pairs(rgb_data, label_data, image_size)

    assert len(images) == len(labels), "Number of images and labels do not match."
    print(f"Number of valid pairs: {len(images)}")
    
    return images, labels