from torch.utils.data import Dataset
import numpy as np
import os
import gc
from numpy.lib.format import open_memmap
from tqdm import tqdm
import torch

class MemmapDataset(Dataset):
    def __init__(self, images: np.memmap, labels: np.memmap, start: int=0, end: int=-1):
        """
        Inputs are expected to be memory mapped numpy arrays (.npy)

        The start and end options allow for a dataset to be viewed as a slice of the original dataset.

        Args:
            images (np.memmap): Memory mapped numpy array of images
            labels (np.memmap): Memory mapped numpy array of labels
            parent_dir (str): Directory where the example and label memmap files are stored
            start (int): Start index for slicing the dataset (default: 0)
            end (int): End index for slicing the dataset (default: None, which means the end of the dataset)
        """
        if end == -1: end = images.shape[0]
        self.start = int(start)
        self.end = int(end)
        assert 0 <= self.start <= self.end <= images.shape[0], "Invalid start/end"

        self.images = images
        self.labels = labels
        self.parent_dir = os.path.dirname(getattr(images, "filename", "")) or "Unknown"

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        self.mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3,1,1)
        self.std_tensor  = torch.tensor(std,  dtype=torch.float32).view(3,1,1)

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx):
        i = self.start + idx
        image = self.images[i].copy()
        label = self.labels[i].copy()

        x = torch.from_numpy(image).float()
        # scale if 0..255
        if x.max() > 1.5: x = x / 255.0

        # expect CHW; if HWC, permute once-per-item
        if x.ndim == 3 and x.shape[0] != 3 and x.shape[-1] == 3:
            x = x.permute(2,0,1)
        x = (x - self.mean_tensor) / self.std_tensor

        return x, torch.as_tensor(label, dtype=torch.long)
    
    def shuffle(self):
        # Check if dataset is read only
        if self.images.flags.writeable == False or self.labels.flags.writeable == False:
            print("Dataset is read-only and cannot be shuffled. To shuffle, reopen the dataset with writable copies (r+).")
            return

        # Shuffle data one entry at a time using Fisher-Yates shuffle
        dataset_size = self.end - self.start

        for i in range(self.end - 1, self.start, -1):
            print(f"Percent Shuffled: {100*(dataset_size-i)/dataset_size:.2f}%", end='\r')
            j = np.random.randint(self.start, i+1)
            self.images[i], self.images[j] = self.images[j], self.images[i]
            self.labels[i], self.labels[j] = self.labels[j], self.labels[i]

            if i % 5000 == 0:
                self.images.flush()
                self.labels.flush()

    def split(self, split_ratio: float):
        n = self.__len__()
        split_n = int(n * split_ratio)
        train  = MemmapDataset(self.images, self.labels, start=self.start,            end=self.start + split_n)
        valid = MemmapDataset(self.images, self.labels, start=self.start + split_n, end=self.end)
        return train, valid
    
    def get_parent_dir(self) -> str:
        """
        Returns the parent directory where the memmap files are stored.
        """
        return self.parent_dir
    
    def save_training_split(self, output_dir: str, example_ext:str="images.npy", label_ext:str="labels.npy", data_split=0.9, slice_size=1024):
        """
        Splits the dataset into training and validation sets and saves them as memmaps.

        Parameters:
            parent_dir (str): Base directory where the examples and labels are stored.
            example_ext (str): Extension for the example files (default: "images.npy").
            label_ext (str): Extension for the label files (default: "labels.npy").
            data_split (float): Proportion of the dataset to use for training (default: 0.9).
            slice_size (int): Size of chunks to read and write (default: 1024).
        """
        ex_shape = self.images.shape
        lb_shape = self.labels.shape
        n_ex, n_lb = ex_shape[0], lb_shape[0]
        assert n_ex == n_lb, f"Count mismatch: {n_ex} images vs {n_lb} labels"

        n_train = int(n_ex * data_split)
        n_valid = n_ex - n_train

        out_train = os.path.join(output_dir, "train"); os.makedirs(out_train, exist_ok=True)
        out_valid = os.path.join(output_dir, "valid"); os.makedirs(out_valid, exist_ok=True)

        # images
        memmap_copy(self.images, os.path.join(out_train, example_ext), source_indices=(0, n_train),          dest_indices=None, chunk=slice_size)
        memmap_copy(self.images, os.path.join(out_valid, example_ext), source_indices=(n_train, n_ex),       dest_indices=None, chunk=slice_size)

        # labels
        memmap_copy(self.labels, os.path.join(out_train, label_ext),  source_indices=(0, n_train),          dest_indices=None, chunk=slice_size)
        memmap_copy(self.labels, os.path.join(out_valid, label_ext),  source_indices=(n_train, n_lb),       dest_indices=None, chunk=slice_size)

        print(f"Train: {n_train}, Valid: {n_valid}")
    
    def concat(self, other: 'MemmapDataset', output_path: str, chunk_size: int = 1024) -> 'MemmapDataset':
        """
        Concatenates another MemmapDataset to the current one and saves it as a new memmap file.
        
        Parameters:
            other (MemmapDataset): The other dataset to concatenate.
            chunk_size (int): Size of chunks to read and write (default: 1024).
            output_path (str): Directory for the output concatenated memmap file.
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if output_path is None:
            raise ValueError("Output path must be specified for concatenation.")
        
        images_path = os.path.join(output_path, "images.npy")
        labels_path = os.path.join(output_path, "labels.npy")

        _memmap_concat(self.images.filename, other.images.filename, chunk_size, images_path)
        _memmap_concat(self.labels.filename, other.labels.filename, chunk_size, labels_path)

        self.images.flush()
        self.labels.flush()
        other.images.flush()
        other.labels.flush()

        return MemmapDataset(
            images=np.load(images_path, mmap_mode='r', allow_pickle=True),
            labels=np.load(labels_path, mmap_mode='r', allow_pickle=True),
        )

    def copy_to(self, output_path: str, source_indices=None, dest_indices=None, chunk_size: int = 1024):
        """
        Copies the memmap dataset to a new location.

        Parameters:
            output_path (str): The directory where the dataset should be copied.
            source_indices (tuple): Indices to copy from the source file (start, end). If None, copies the entire file.
            dest_indices (tuple): Indices to write to the destination file (start, end). If None, writes to the beginning.
            chunk_size (int): Number of elements to copy at a time.
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Locate source and destination indices
        if source_indices is None:
            source_indices = self.start, self.end
        else:
            source_indices = (source_indices[0] + self.start, source_indices[1] + self.start)
        if dest_indices is None:
            dest_indices = (0, source_indices[1] - source_indices[0])

        if dest_indices[1] - dest_indices[0] != source_indices[1] - source_indices[0]:
            raise ValueError(f"Source and destination sizes must match. Got source: {source_indices} and destination: {dest_indices}")

        memmap_copy(self.images, os.path.join(output_path, "images.npy"), source_indices, dest_indices, chunk_size)
        memmap_copy(self.labels, os.path.join(output_path, "labels.npy"), source_indices, dest_indices, chunk_size)

def _memmap_concat(memmap1, memmap2, chunk_size, output_path):
    """
    Concatenate two memmaps into a new memmap file.
    Parameters:
        memmap1 (str): Path to the first memmap file.
        memmap2 (str): Path to the second memmap file.
        chunk_size (int): Size of chunks to read and write.
        output_path (str): Path for the output concatenated memmap file.
    """
    a = np.load(memmap1, mmap_mode='r')
    b = np.load(memmap2, mmap_mode='r')

    
    if a.shape[1:] != b.shape[1:]: raise ValueError(f"{a.shape} vs {b.shape}")
    if a.dtype != b.dtype: raise ValueError(f"{a.dtype} vs {b.dtype}")

    out = open_memmap(output_path, mode='w+', dtype=a.dtype,
                      shape=(a.shape[0]+b.shape[0],) + a.shape[1:])

    print(f"\nConcatenating to {output_path}...")

    print(f"Array 1: {a.shape}, Array 2: {b.shape}")
    # copy A
    for i in tqdm(range(0, a.shape[0], chunk_size), desc=f"Copying Array 1"):
        j = min(i+chunk_size, a.shape[0])
        np.copyto(out[i:j], a[i:j], casting='no')
        out.flush()
        del a
        a = np.load(memmap1, mmap_mode='r')  # Reload to avoid memory issues

    # copy B
    off = a.shape[0]
    for i in tqdm(range(0, b.shape[0], chunk_size), desc=f"Copying Array 2"):
        j = min(i+chunk_size, b.shape[0])
        np.copyto(out[off+i:off+j], b[i:j], casting='no')
        out.flush()
        del b
        b = np.load(memmap2, mmap_mode='r')  # Reload to avoid memory issues

    print(f"Resulting output array shape: {out.shape}, dtype: {out.dtype}")
    del a, b, out


def memmap_copy(src: np.memmap, dst_path: str, source_indices=None, dest_indices=None, chunk=1024):
    """    
    Copy a range of elements from a source memmap file to an output location.

    Parameters:
        src (np.memmap): Source memmap object.
        dst_path (str): Path to the destination.
        source_indices (tuple): Indices to copy from the source file (start, end). If None, copies the entire file.
        dest_indices (tuple): Indices to write to the destination file (start, end). If None, writes to the beginning.
        chunk (int): Number of elements to copy at a time.
    """
    dtype = src.dtype
    src_shape = src.shape

    start, end = (0, src_shape[0]) if not source_indices else source_indices
    dst_start, dst_end = (0, end-start) if not dest_indices else dest_indices

    # Open destination memmap
    if not os.path.exists(dst_path):
        full_shape = (dst_end-dst_start,) + src_shape[1:]
        dst = open_memmap(dst_path, mode='w+', dtype=dtype, shape=full_shape)
    else:
        dst = open_memmap(dst_path, mode='r+')

    # Copy chunk-by-chunk with fresh opens to minimize OS caching
    write_pos = dst_start
    for i in tqdm(range(start, end, chunk), desc=f"Copying {os.path.basename(src.filename) if isinstance(src.filename, str) else ' '}"):
        j = min(i + chunk, end)
        dst_end_pos = j-i

        dst = open_memmap(dst_path, mode='r+')
        np.copyto(dst[write_pos:write_pos + dst_end_pos], src[i:j], casting='no')

        dst.flush()
        src.flush()
        del dst
        write_pos += dst_end_pos

    gc.collect()