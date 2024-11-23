import rasterio
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from joblib import Parallel, delayed, dump
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = None   # Otherwise can't open large images

def main():
    mask_filepath = "../dataset/ortho_mask.tif"
    out_mask_filepath_jpg = "../dataset/ortho_mask_gaussian.jpg"
    out_mask_filepath_tif = "../dataset/ortho_mask_gaussian.tif"
    val_mask_filepath = "../dataset/val_mask.png"
    # height and width of gaussian kernel
    # lower bound for thresholding after blur (anything greater is 1) 

    # Reading mask (.tif)
    with rasterio.open(mask_filepath) as src:
        mask = src.read(1)      # Reading first band
        metadata = src.meta     # Getting metadata (geospatial info)
    
    # Reading validation mask
    val_mask = Image.open(val_mask_filepath)
    val_mask = np.array(val_mask)
    val_mask = val_mask[:mask.shape[0],0:mask.shape[1],0]   # Cropping original image to pred mask size
    val_mask = (val_mask > 250) * 255

    # Set True to search, false for manual parameter tuning
    search = False
    if search == True:
        # Searching parameter space
        kernel_sizes = range(1,101,2)               # height and width of gaussian kernel
        lower_bounds = np.linspace(240,255,20)      # lower bound for thresholding after blur (anything greater is 1) 

        # Creating parameter combinations list
        param_list = create_param_list(kernel_sizes, lower_bounds)

        # Creating memmaps
        print("Creating memmaps...")
        dump(mask, "mask.z")
        dump(val_mask, "val_mask.z")
        mask = np.memmap("mask.z", mode="r+", shape=mask.shape)
        val_mask = np.memmap("val_mask.z", mode="r+", shape=val_mask.shape)

        # Testing parameters
        print("Searching parameter space...")
        results = Parallel(n_jobs=12, mmap_mode="r")(delayed(test_params_parallel)
            (param_pair, mask, val_mask) for param_pair in tqdm(param_list))

        # Getting optimal params
        best_kernel_size = 0
        best_lower_bound = 0
        best_IOU = 0
        for result in results:
            if result[0] > best_IOU:
                best_IOU = result[0]
                best_kernel_size = result[1]
                best_lower_bound = result[2]

        print(f"Best IOU: {best_IOU}")
        print(f"Best Kernel Size: {best_kernel_size}")
        print(f"Best Lower Bound: {best_lower_bound}")

        # Cleanup
        os.remove("mask.z")
        os.remove("val_mask.z")

        # For single example
        best_mask = filter_image(mask, best_kernel_size, best_lower_bound).astype('uint8')
        display_hist(best_mask)
        print(best_mask)
    else:
        # Change as needed
        kernel_size = 99
        lower_bound = 250
        best_mask = filter_image(mask, kernel_size, lower_bound).astype('uint8')

    # Saving output mask (.jpg)
    im = Image.fromarray(best_mask)
    im.save(out_mask_filepath_jpg)

    # Saving output mask (.tif)
    with rasterio.open(out_mask_filepath_tif, "w", **metadata) as dest:
        # Rasterio needs [bands, width, height]
        best_mask = np.rollaxis(best_mask, 1)
        dest.write(best_mask[np.newaxis,...])

def threshold(array, lower_bound):
    return (array > lower_bound) * 255

# Show image pixel intensity distribution, to be used after applying a filter 
# to manually figure out threshold
def display_hist(array):
    plt.hist(array.ravel(), bins=256, range=(0.0, 255.0), fc='k', ec='k')
    plt.show()

# Tests set of parameters, returns metric
def test_params(array, val_array, kernel_size, lower_bound):
    new_array = cv2.GaussianBlur(array,(kernel_size,kernel_size),0)
    new_array = threshold(new_array, lower_bound)
    return IOU(new_array, val_array)
    
def test_params_parallel(param_pair, array, val_array):
    kernel_size = param_pair[0]
    lower_bound = param_pair[1]
    new_array = cv2.GaussianBlur(array,(kernel_size,kernel_size),0)
    new_array = threshold(new_array, lower_bound)
    return (IOU(new_array, val_array), kernel_size, lower_bound)

# Processes image using parameters
def filter_image(array, kernel_size, lower_bound):
    new_array = cv2.GaussianBlur(array,(kernel_size,kernel_size),0)
    new_array = threshold(new_array, lower_bound)
    return new_array

def IOU(array, val_array):
    assert array.shape == val_array.shape, "Array and Validation Array have different sizes."
    intersection = np.sum(array == val_array)
    union = array.shape[0] * array.shape[1]
    return intersection / union

def create_param_list(param1, param2):
    param_list = []
    for p1 in param1:
        for p2 in param2:
            param_list.append((p1, p2))
    return param_list


def create_param_grid(param1, param2):
    #grid = np.zeros((len(param1), len(param2))
    grid = []
    for y in range(len(param1)):
        row = []
        for x in range(len(param2)):
            param_pair = (param1[y], param2[x])
            row.append(param_pair)
            #grid[y,x] = (param1[y], param2[x])
        grid.append(row)
    return grid

def create_idx_list(y_len, x_len):
    idx_list = []
    for x in range(x_len):
        for y in range(y_len):
            idx_list.append((x,y))
    return idx_list
    


if __name__ == "__main__":
    main()