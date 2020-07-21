import os 
from os.path import join, isdir, isfile
from tqdm import tqdm
import shutil

count = 0


for folder in tqdm(os.listdir(os.getcwd())):

    print(folder)

    if isdir(join(folder,"0")):
        for file in tqdm(os.listdir(join(folder,"0"))):
            if file.endswith(".tif"):
                hr_file = join(folder,"0_hr","hr_" + file)
                lr_file = join(folder,"0", file)

                lr_dest = join("dataset", "0", "lr_" + str(count) + ".tif")
                hr_dest = join("dataset", "0", "hr_" + str(count) + ".tif")

                if isfile(hr_file) and isfile(lr_file):
                    shutil.move(hr_file, hr_dest)
                    shutil.move(lr_file, lr_dest)
                    count += 1
                    
            


            

    if isdir(join(folder,"1")):
        for file in tqdm(os.listdir(join(folder,"1"))):
            if file.endswith(".tif"):
                hr_file = join(folder,"1_hr","hr_" + file)
                lr_file = join(folder,"1", file)
                
                lr_dest = join("dataset", "1", "lr_" + str(count) + ".tif")
                hr_dest = join("dataset", "1", "hr_" + str(count) + ".tif")
                
                if isfile(hr_file) and isfile(lr_file):
                    shutil.move(hr_file, hr_dest)
                    shutil.move(lr_file, lr_dest)
                    count += 1

print(count)
