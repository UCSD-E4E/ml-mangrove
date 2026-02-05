import argparse
import os
import subprocess
import numpy as np
from tqdm import tqdm
from PIL import Image
parser = argparse.ArgumentParser(description="Retile an orthomosiac, call this in the same folder as the orthomosaic")
parser.add_argument("--width",help = "Width of output tiles")
parser.add_argument("--input", help = "input orthomosaic")
parser.add_argument("--targetDir", help = "output directory of tile imags with respect to the current directory")
parser.add_argument("--shpFile", help = "input shapefile of polygon that you want to clip the orthomosaic to")
args = parser.parse_args()

if args.width:
	out_width = args.width
if args.input:
	img_input = args.input
if args.targetDir:
	outputDir = args.targetDir
if args.shpFile:
	polygon = args.shpFile
else:
	polygon = False

if not img_input.lower().endswith('.tif'):
	print("Input raster is not of .tif format")
	exit()

cwd = os.getcwd()
if os.path.exists(os.path.join(cwd,outputDir)) == False:
	os.mkdir(os.path.join(cwd,outputDir))

if isinstance(polygon, str):
    if not polygon.lower().endswith('.shp'):
        print(polygon)
        print("Input polygon is not of type .shp format")
        exit() 
    #polycall = "gdal -clip " + polygon + " " + img_input + " " + "clipped" + img_input
    polycall = "gdalwarp " + "-cutline " + polygon + " -dstalpha -crop_to_cutline " + img_input  + " " + "clipped"+img_input  
    subprocess.call(polycall, shell=True)
    call = "gdal_retile.py -ps " + out_width + " " + out_width + " " + "-targetDir " + outputDir + " clipped" + img_input
    subprocess.call(call, shell=True)


else:
    call = "gdal_retile.py -ps " + out_width + " " + out_width + " " + "-targetDir " + outputDir + " " + img_input
    print(call)
    subprocess.call(call, shell=True)

img_dir = os.path.join(cwd, outputDir)
print("Removing undersized tiles in:")
print(img_dir)
for filename in tqdm(os.listdir(img_dir)):
	filepath = os.path.join(img_dir, filename)
	if os.path.splitext(filename)[1] == ".tif":
		with Image.open(filepath) as im:
			x, y = im.size
			totalsize = x*y
			totalsum = np.sum(np.array(im))
		if totalsize < (int(out_width) * (int(out_width))):
			os.remove(filepath)
		elif np.array_equal(np.unique(np.array(im)), [0, 255]):
			os.remove(filepath)
		elif np.count_nonzero(np.array(im) == np.array([0,0,0,0])) > (int(out_width) * (int(out_width))/30):
			os.remove(filepath)