# Example usage: python3 gen_seg_labels.py --width 256 --input_raster test_data/test.tif --input_mask test_data/masks/mask_binary.tif -c -d
# Or on a .shp file: python3 gen_seg_labels.py --width 256 --input_raster test_data/test.tif --input_vector test_data/test.shp

import argparse
import os
import subprocess
import numpy as np
from tqdm import tqdm
from PIL import Image
from glob import glob
from joblib import Parallel, delayed
from raster_mask import raster_mask

def remove_undersized_tiles(img_filename, img_dir, label_dir, img_file_basename, mask_file_basename, out_width):
	img_filepath = os.path.join(img_dir, img_filename)
	index = img_filename.replace(img_file_basename, '')
	label_filename = mask_file_basename + index
	label_filepath = os.path.join(label_dir, label_filename)
	if os.path.splitext(img_filename)[1] == ".tif":
		with Image.open(img_filepath) as im:
			x, y = im.size
			totalsize = x*y
			totalsum = np.sum(np.array(im))
		if totalsize < (int(out_width) * (int(out_width))):
			os.remove(img_filepath)
			os.remove(label_filepath)
		elif np.array_equal(np.unique(np.array(im)), [0, 255]):
			os.remove(img_filepath)
			os.remove(label_filepath)

def tif_to_jpg(file, destructive):
	with Image.open(file) as im:
		new_im = im.convert("RGB")
		new_file = file.rstrip(".tif")
		new_im.save(new_file + ".jpg", "JPEG")
	if (destructive == True):
		os.remove(file)
	

def gen_seg_labels(out_width, raster_file, vector_file, mask_file, out_dir, convert, destructive):
	# Check
	if not raster_file.lower().endswith('.tif'):
		print("Input raster is not of .tif format")
		exit()

	# Useful definitions
	img_file_basename = os.path.basename(raster_file).split('.')[0]
	mask_file_basename = "mask_binary"

	# Creating necessary directories
	img_dir = os.path.join(out_dir, "images")
	label_dir = os.path.join(out_dir, "labels")
	if os.path.exists(img_dir):
		os.system("rm -r " + img_dir)
	if os.path.exists(label_dir):
		os.system("rm -r " + label_dir)
	os.mkdir(img_dir)
	os.mkdir(label_dir)

	# Tiling
	print()
	print("Executing GDAL calls...")
	call = "gdal_retile.py -ps " + out_width + " " + out_width + " " + "-targetDir " + img_dir + " " + raster_file
	print(call)
	subprocess.call(call, shell=True)
	call = "gdal_retile.py -ps " + out_width + " " + out_width + " " + "-targetDir " + label_dir + " " + mask_file
	print(call)
	subprocess.call(call, shell=True)

	# Removing undersized and empty tiles (in img and label directories)
	print("Removing undersized tiles...")
	with Parallel(n_jobs=-1) as parallel:
		parallel(delayed(remove_undersized_tiles)
				(img_filename, img_dir, label_dir, img_file_basename, mask_file_basename, out_width) 
				for img_filename in tqdm(os.listdir(img_dir)))

		print("Number of Images: " + str(len(os.listdir(img_dir))))
		print("Number of Labels: " + str(len(os.listdir(label_dir))))

		# Converting from tif to jpg
		if convert == True:
			print("Converting images from .tif to .jpg")
			parallel(delayed(tif_to_jpg)(file, destructive) for file in tqdm(glob(os.path.join(img_dir, "*.tif"))))
			print("Converting labels from .tif to .jpg")
			parallel(delayed(tif_to_jpg)(file, destructive) for file in tqdm(glob(os.path.join(label_dir, "*.tif"))))

	print("Creating Map...")
	map_filepath = os.path.join(out_dir, "map.txt")
	with open(map_filepath, 'w') as map_file:
		for img_filename in tqdm(os.listdir(img_dir)):
			if img_filename.split('.')[1] == "tif":
				continue
			img_filepath = os.path.join(img_dir, img_filename)
			img_filepath_abs = os.path.abspath(img_filepath)
			index = img_filename.replace(img_file_basename, '')
			label_filename = mask_file_basename + index
			label_filepath = os.path.join(label_dir, label_filename)
			label_filepath_abs = os.path.abspath(label_filepath)
			# NOTE: Consider adding parser for delimiter in map file
			map_file.write(img_filepath_abs + " -> " + label_filepath_abs + "\n")

	print("Done.")

def tile_raster(out_width, raster_file, out_dir, convert, destructive):
	# Check
	if not raster_file.lower().endswith('.tif'):
		print("Input raster is not of .tif format")
		exit()

	# Useful definitions
	img_file_basename = os.path.basename(raster_file).split('.')[0]

	# Creating necessary directories
	img_dir = os.path.join(out_dir, "images")
	if os.path.exists(img_dir):
		os.system("rm -r " + img_dir)
	os.mkdir(img_dir)

	# Tiling
	print()
	print("Executing GDAL calls...")
	call = "gdal_retile.py -ps " + out_width + " " + out_width + " " + "-targetDir " + img_dir + " " + raster_file
	print(call)
	subprocess.call(call, shell=True)

	# Will use this modified function for removing undersized tiles
	def remove_undersized_tiles2(img_filename, img_dir, img_file_basename, out_width):
		img_filepath = os.path.join(img_dir, img_filename)
		if os.path.splitext(img_filename)[1] == ".tif":
			with Image.open(img_filepath) as im:
				x, y = im.size
				totalsize = x*y
				totalsum = np.sum(np.array(im))
			if totalsize < (int(out_width) * (int(out_width))):
				os.remove(img_filepath)
			elif np.array_equal(np.unique(np.array(im)), [0, 255]):
				os.remove(img_filepath)

	# Removing undersized and empty tiles (in img and label directories)
	print("Removing undersized tiles...")
	with Parallel(n_jobs=-1) as parallel:
		parallel(delayed(remove_undersized_tiles2)
				(img_filename, img_dir, img_file_basename, out_width) 
				for img_filename in tqdm(os.listdir(img_dir)))

		print("Number of Images: " + str(len(os.listdir(img_dir))))
		# Converting from tif to jpg
		if convert == True:
			print("Converting images from .tif to .jpg")
			parallel(delayed(tif_to_jpg)(file, destructive) for file in tqdm(glob(os.path.join(img_dir, "*.tif"))))

	print("Done.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Tile and orthomosaic and it's corresponding mask file and create label mapping")
	parser.add_argument("--width",help = "Width of output tiles")
	parser.add_argument("--input_raster", help = "input orthomosaic (.tif)")
	parser.add_argument("--input_mask", help = "input (binary) mask (.tif)")
	parser.add_argument("--input_vector", help = "Only necessary if input_mask is not specified, should be a labeled .shp file")
	parser.add_argument("--out_dir", help = "location to create directories for tiles and labels (defaults to directory containing raster)")
	parser.add_argument("-c", action='store_true', help = "Automatically convert resulting .tif files to .jpg (no argument)")
	parser.add_argument("-d", action='store_true', help = "Destructive conversion from .tif to .jpg (Removes the .tif file)")
	args = parser.parse_args()

	if args.width:
		out_width = args.width
	else:
		print("Need to specify width, exiting.")
		exit()
	if args.input_raster:
		raster_file = args.input_raster
	else:
		print("Need to specify raster file, exiting.")
		exit()
	if args.input_vector:
		vector_file = args.input_vector
	else:
		if not args.input_mask:
			print("Need to specify input vector or input mask, exiting.")
			exit()
		else:
			vector_file = None
	if args.input_mask:
		mask_file = args.input_mask
	else:
		# Creating masks if they don't exist already
		print("Creating raster_masks...")
		raster_mask(raster_file, vector_file)
		temp_dir = os.path.dirname(raster_file)
		mask_file = os.path.join(temp_dir, "masks", "mask_binary.tif")
	if args.out_dir:
		out_dir = args.out_dir
	else:
		out_dir = os.path.dirname(raster_file)
	if args.c:
		convert = True
	else:
		convert = False
	if args.d:
		destructive = True
	else:
		destructive = False

	gen_seg_labels(out_width, raster_file, vector_file, mask_file, out_dir, convert, destructive)
