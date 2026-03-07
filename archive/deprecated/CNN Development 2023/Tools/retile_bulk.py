import argparse
import os
import sys
import subprocess
import numpy as np
from tqdm import tqdm
from PIL import Image
 
 
 
def get_tif(dir):
    tifs = []
    for path, _, files in os.walk(dir):
        for name in files:
            file_name = os.path.join(path, name)
            if file_name.endswith('.tif'):
                #If the file is actually an orthomosaic and not an existing tile
                if os.path.getsize(file_name) > 50000000:
                    tifs.append(file_name)
    return tifs
 
def get_shp(file):
    vectors = []
   
    cur_path = os.path.dirname(file)
    for path, _, files in os.walk(cur_path):
        for name in files:
            file_name = os.path.join(path, name)
            if name.endswith('.shp') and (name.startswith('0') or name.startswith('1')):
                vectors.append(file_name)
    return vectors, cur_path
 
 
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retile an orthomosiac, call this in the same folder as the orthomosaic")
    parser.add_argument("--width",help = "Width of output tiles")
    args = parser.parse_args()
    if args.width:
        width = args.width
    else:
        print("No output width was given, check --width argument")
        sys.exit()
   
 
    cwd = os.getcwd()
    images = get_tif(cwd)
    for image in images:
        shp_files, path  = get_shp(image)
        for file in shp_files:
            cur_path = os.path.splitext(file)[0]
            base=os.path.splitext(os.path.basename(file))[0]
            if os.path.exists(cur_path) == False:
                os.mkdir(cur_path)
            new_str = base + "_clipped_" + os.path.basename(image)
            image_output = os.path.join(cur_path, new_str)
           
            #Clipping orthomosaics to label size
            print("Clipping ortho to label size\n")
            polycall = "gdalwarp " + "-cutline \"" + file + "\" -dstalpha -crop_to_cutline \"" + image  + "\" " + "\"" + image_output + "\""
            print(polycall)

            subprocess.call(polycall, shell=True)
            print("Tilizing clipped ortho\n")
            call = "gdal_retile.py -ps " + width + " " + width + " " + "-targetDir \"" + cur_path + "\" " + "\"" + image_output + "\""
            print(call)
            subprocess.call(call, shell=True)
 
            print("Removing undersized tiles in:")
            print(cur_path)
 

            for filename in tqdm(os.listdir(cur_path)):
                filepath = os.path.join(cur_path, filename)
                if os.path.getsize(filepath) < 50000000: 
                    if os.path.splitext(filename)[1] == ".tif":
                        with Image.open(filepath) as im:
                            x, y = im.size
                            totalsize = x*y
                            totalsum = np.sum(np.array(im))
                        if totalsize < (int(width) * (int(width))):
                            os.remove(filepath)
                        elif np.array_equal(np.unique(np.array(im)), [0, 255]):
                            os.remove(filepath)
                        elif totalsum == 0:
                            os.remove(filepath)
 
       
       
 
   
 
