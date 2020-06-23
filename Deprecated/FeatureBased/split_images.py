'''
Splits a large directory of images into directories with fewer images in order to view thumbnails with less
overhead.
'''

import shutil, os, sys
from tqdm import tqdm, trange

path = '/home/sam/Downloads/MVNM_Training_Data_128'
files_per_dir = 20000
for d in os.listdir(path):
    files = os.listdir(os.path.join(path, d))
    for i in range((len(files)//files_per_dir) + 1):
        d_name = '{}{}'.format(d, i)
        if not os.path.isdir(os.path.join(path, d_name)):
            os.mkdir(os.path.join(path, d_name))
        for j in trange(i*files_per_dir, min((i+1)*files_per_dir, len(files))):
            src = os.path.join(path, d, files[j])
            dest = os.path.join(path, d_name, files[j])
            shutil.move(src, dest)