import os 
import geopandas as gpd
from os.path import join, split
import rasterio 
from shapely.geometry import box
from fiona.crs import from_epsg
from tqdm import tqdm
from rasterio import features
import cv2
from rasterio.mask import mask
from rasterio.plot import reshape_as_raster, reshape_as_image
from rasterio.enums import Resampling
from rasterio import Affine
import numpy as np
hr_size = 128
lr_size = 64



big_img = rasterio.open("lr_src.tif")

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]
        
#getting image folder
for image in tqdm(os.listdir(join(os.getcwd(), "tiles"))):
    img_loc = join(os.getcwd(), "tiles", image)
    hr_loc = join(os.getcwd(), "hr_tiles", image)
    lr_loc = join(os.getcwd(), "lr_tiles", "lr_" + image)
    #getting images

    if image.endswith(".tif"):
        with rasterio.open(img_loc) as cur_img:

            # resample data to target shape
            array = cur_img.read(out_shape=(cur_img.count, int(hr_size), int(hr_size)), resampling=Resampling.bilinear)[:3,:,:]


            t = cur_img.transform
            scale = cur_img.height / hr_size
            # rescale the metadata
            transform = Affine(t.a * scale, t.b, t.c, t.d, t.e * scale, t.f)
            height = int(cur_img.height / scale)
            width = int(cur_img.width / scale)
            # scale image transform
            out_meta = cur_img.meta.copy()

            out_meta.update({"count":3, "transform": transform, "height": height, "width": width})
        
            # writing new image
            with rasterio.open(hr_loc, "w", **out_meta) as dest:
                dest.write(array)
        
            # getting bounds of original tile
            bounds = cur_img.bounds
            bbox = box(cur_img.bounds.left, cur_img.bounds.bottom, cur_img.bounds.right, cur_img.bounds.top)
            geo = gpd.GeoDataFrame({'geometry': bbox}, index = [0])
            coords = getFeatures(geo)
            try:
                #masking large image to tile
                out_img, out_transform = mask(big_img, coords, crop = True)
                out_meta = cur_img.meta.copy()
                print(out_img.shape)
                scale = cur_img.shape[1] / lr_size
                nout_transform = Affine(out_transform.a * scale, out_transform.b, out_transform.c, out_transform.d, out_transform.e * scale, out_transform.f)

                out_meta.update({"count":3, "height": lr_size, "width": lr_size, "transform":nout_transform})
                if np.count_nonzero(reshape_as_image(out_img) == np.array([0,0,0,0])) < ((lr_size*lr_size)/30):
                    with rasterio.open(lr_loc, "w", **out_meta) as dest:
                        dest.write(out_img[:3,:,:])
                else:
                    os.remove(hr_loc)
            except Exception as e:
                print(e)
                os.remove(hr_loc)

            