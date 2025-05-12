import rasterio
import os
import shutil
import geopandas as gpd
from pyproj import CRS
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from shapely.geometry import shape, Polygon, MultiPolygon

"""
This script resamples TIFF files to a specified resolution and rasterizes/aligns shapefiles to the resampled TIFFs.

It expects each TIFF file to be in a directory structure where each TIFF is in its own folder, along with its label files in a subfolder named "labels".
"""


def get_utm_crs(lon, lat):
    utm_zone = int((lon + 180) / 6) + 1
    is_northern = lat >= 0
    return CRS.from_dict({
        "proj": "utm",
        "zone": utm_zone,
        "datum": "WGS84",
        "south": not is_northern
    })

def align_shapefiles(output_root,
    simplify_tolerance=1.0,
    buffer_amount=0.5,
    min_area=10.0
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
        if has_labels: 
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

def resample_tiff(input_root, output_root, target_resolution, align_shapefile=True):
    if not os.path.isdir(input_root):
        raise NotADirectoryError(f"{input_root} is not a valid directory.")
    
    # Resample TIFFs
    for root, _, files in os.walk(input_root):
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
                    dst_crs = get_utm_crs(lon, lat)

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
    
    if align_shapefile:
        align_shapefiles(output_root)

def __main__():
    input_root = "/Path/To/Your/Drone/Data"
    output_root = "/Path/To/Your/Output/Data"
    target_resolution = 1.0 # Resolution in meters
    resample_tiff(input_root, output_root, target_resolution, align_shapefiles=True)

if __name__ == "__main__":
    __main__()