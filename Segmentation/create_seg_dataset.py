# Example usage: python3 create_seg_dataset.py --map_files ../dataset/Site_1/map.txt ../dataset/Site_4/map.txt ../dataset/Site_5/map.txt

import argparse
import os
from shutil import copyfile

def create_seg_dataset(map_files, dir_name, include_tif):
    # Note this file structure must exist (or you can change the paths)
    # TODO add parser arguments for flexibility
    img_dir = f"../dataset/{dir_name}/images"
    ann_dir = f"../dataset/{dir_name}/annotations"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(ann_dir):
        os.makedirs(ann_dir)

    delimeter = " -> "
    pair_id = 0

    for map_file in map_files:
        with open(map_file) as mf:
            while True:
                line = mf.readline()

                # breaks when EOF is reached
                if not line:
                    break
                
                # reading image and annotation paths
                img_path, ann_path = line.rstrip().split(sep=delimeter)

                # generating destination paths
                img_dest = os.path.join(img_dir, "image_{}.jpg".format(pair_id))
                ann_dest = os.path.join(ann_dir, "annotation_{}.jpg".format(pair_id))

                # copying files
                copyfile(img_path, img_dest)
                copyfile(ann_path, ann_dest)

                # if .tif files are to be moved as well
                if include_tif == True:
                    tif_path = img_path.replace("jpg", "tif")
                    tif_dest = os.path.join(img_dir, "image_{}.tif".format(pair_id))
                    copyfile(tif_path, tif_dest)

                pair_id += 1 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset using map files")
    parser.add_argument("--map_files", nargs='+', help = "space separated map file paths")
    parser.add_argument("--tif", action="store_true", help = "add if you want dataset to contain tifs as well")
    parser.add_argument("--train", action="store_true", help = "creating a training dataset")
    parser.add_argument("--test", action="store_true", help = "creating a testing dataset")
    args = parser.parse_args()
    
    if args.map_files:
        map_files = args.map_files
    else:
        print("Need to specify map files, exiting.")
        exit()
    if args.tif:
        include_tif = True
    else:
        include_tif = False
    if args.train and args.test:
        print("Pick either train or test, not both. Exiting.")
        exit()
    elif args.train:
        dir_name = "training"
    elif args.test:
        dir_name = "testing"
    else:
        print("Need to specify if creating a training set or testing set")
        exit()
    
    create_seg_dataset(map_files, dir_name, include_tif)