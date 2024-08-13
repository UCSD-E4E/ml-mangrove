import os
import re
import shutil

"""
Data preprocessing script for the Mangrove dataset that standardizes the filenames.
You still need to manually go through the dataset and clean a lot of the data due to inconsistencies.

This assumes that the data is organized in the following way:
- A directory called "mangrove_data" containing:
- A directory called "Chunks" containing directories named "Chunk 1", "Chunk 2", ..., "Chunk 22"
- A directory called "Labels" containing directories named "Chunk 1 labels", "Chunk 2 labels", ..., "Chunk 22 labels"
"""
def rename_tifs(base_dir: str, chunk_num):
    chunk_dir = os.path.join(base_dir, f"Chunk {chunk_num}")
    for root, dirs, files in os.walk(chunk_dir):
        for file in files:
            if file.endswith(".tif"):
                # Use regex to extract the chunk number and the last two numbers
                match = re.match(r"^.*-(\d+)-(\d+)\.tif$", file)
                match2 = re.match(r"^[^-]*-(\d+)-(\d+).*\.tif$", file)
                match3 = re.match(r"^.*rtho.tif$", file)
                if match:
                    num1, num2 = match.groups()
                    new_name = f"Chunk{chunk_num}_{num1}-{num2}.tif"
                elif match2:
                    num1, num2 = match2.groups()
                    new_name = f"Chunk{chunk_num}_{num1}-{num2}.tif"
                elif match3:
                    new_name = f"Chunk{chunk_num}.tif"
                elif file.endswith(f"{chunk_num}.tif") and not re.match(r"^.*(\d+)-(\d+)\.tif$", file):
                    new_name = f"Chunk{chunk_num}.tif"
                else:
                    continue
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, new_name)
                try:
                    os.rename(old_path, new_path)
                except FileExistsError:
                    pass

def rename_shapes(chunk_dir: str, chunk_num: str):
    chunk_dir = os.path.join(chunk_dir, f"Chunk {chunk_num}")
    for root, dirs, files in os.walk(chunk_dir):
        for file in files:
            if file.endswith(".shp") and not file.startswith("._"):
                # Use regex to extract the chunk number and the last two numbers
                match = re.match(r"^.*?-(\d+)-(\d+)*\.shp$", file)
                match1 = re.match(r"^.*?(\d+)-(\d+)\.shp$", file)
                match2 = re.match(r"^.*?(\d+).(\d+)*\.shp$", file)
                match3 = re.search(r"(\d+)$", file)
                if match:
                    num1, num2 = match.groups()
                    new_name = f"Chunk{chunk_num}_labels{num1}-{num2}.shp"
                elif match1:
                    num1, num2 = match1.groups()
                    new_name = f"Chunk{chunk_num}_labels{num1}-{num2}.shp"  
                elif match2:
                    num1, num2 = match2.groups()
                    new_name = f"Chunk{chunk_num}_labels{num1}-{num2}.shp"
                elif match3:
                    num1 = match3.groups()[0]
                    new_name = f"Chunk{num1}_labels.shp"
                else:
                    continue
                print(f"Renaming {file} to {new_name}")
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, new_name)
                try:
                    os.rename(old_path, new_path)
                except FileExistsError:
                    pass

def delete_empty_dirs(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                os.rmdir(dir_path)
            except OSError:
                pass  # Directory is not empty or failed to delete

def move_labels_to_chunks(base_dir: str):
    label_dir = os.path.join(base_dir, "Labels")
    chunk_dir = os.path.join(base_dir, "Chunks")
    for root, dirs, files in os.walk(label_dir):
        for dir in dirs:
            if dir.endswith("labels"):
                chunk_num = dir.split(" ")[1]
                source = os.path.join(label_dir, dir)
                destination = os.path.join(chunk_dir, f"Chunk {chunk_num}")
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                print(f"Moving {dir} to {destination}")
                shutil.move(source, destination)

def pair_tifs_and_shapes(base_dir: str):
    move_labels_to_chunks(base_dir)
    chunk_dir = os.path.join(base_dir, "Chunks")
    for root, dirs, files in os.walk(chunk_dir):
        # Get all chunk x directories
        for dir in dirs:
            chunk_subdir = os.path.join(chunk_dir, dir)
            
            chunk_num = dir.split(" ")
            if len(chunk_num) <= 1:
                return
            chunk_num = chunk_num[1]

            tif_files = []
            for _, _, files2 in os.walk(chunk_subdir):
                # Get all tifs in the chunk x directory
                for file in files2:
                    if file.endswith(".tif"):
                        tif_files.append(file)
            # Process tifs and corresponding shape labels within chunk x directory
            for file in tif_files:
                match = re.match(r"^.*(\d+)-(\d+)\.tif$", file)
                match2 = re.match(r"^.*?(\d+\.\d+)\.tif$", file)
                match3 = re.match(r"^.*?(\d+\.\d+)_1.tif$", file)
                match4 = re.match(r"^.*?(\d+)_1\.tif$", file)
                match5 = re.match(r"^.*?(\d+).tif$", file)
                if match:
                    num1, num2 = match.groups()
                    identifier = f"{num1}-{num2}"
                elif match2:
                    num1 = match2.groups()[0]
                    identifier = f"{num1}"
                elif match3:
                    num1 = match3.groups()[0]
                    rename = f"Chunk{num1}.tif"
                    os.rename(os.path.join(chunk_subdir, file), os.path.join(chunk_subdir, rename))
                    file = rename
                    identifier = f"{num1}"
                elif match4:
                    num1 = match4.groups()[0]
                    rename = f"Chunk{num1}.tif"
                    os.rename(os.path.join(chunk_subdir, file), os.path.join(chunk_subdir, rename))
                    file = rename
                    identifier = f"{num1}"
                elif match5:
                    num1 = match5.groups()[0]
                    identifier = f"{num1}"
                else:
                    continue
                # Make new directory
                new_dir = os.path.join(chunk_dir, dir, f"{identifier}")
                os.makedirs(new_dir, exist_ok=True)
                try:
                    # move tif and shape to new directory
                    os.rename(os.path.join(chunk_dir, dir, file), os.path.join(chunk_dir, dir, new_dir, file))
                    if match:
                        label_file = f"Chunk{chunk_num}_labels{identifier}.shp"
                        os.rename(os.path.join(chunk_dir, dir, f"Chunk {chunk_num} labels", label_file), os.path.join(chunk_dir, dir, new_dir, label_file))
                    else: 
                        label_file = f"Chunk{identifier}_labels.shp"
                        os.rename(os.path.join(chunk_dir, dir, f"Chunk {identifier} labels", label_file), os.path.join(chunk_dir, dir, new_dir, label_file))
                except FileNotFoundError:
                    continue

base_dir = 'Users/gage/Desktop/mangrove_data'
chunk_dir = os.path.join(base_dir, "Chunks")
label_dir = os.path.join(base_dir, "Labels")
for i in range(1, 23):
    rename_tifs(chunk_dir, i)
    rename_shapes(chunk_dir, str(i))
for i in range(1, 8):
    num = f"12.{i}"
    num = float(num)
    rename_tifs(chunk_dir, num)
pair_tifs_and_shapes(base_dir)
delete_empty_dirs(base_dir)

