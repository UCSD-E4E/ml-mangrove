from torch.utils.data import Dataset, Sampler, BatchSampler, DataLoader
import numpy as np
from typing import Optional, List, Tuple
import torch

class MemmapDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, validation_indices: Optional[Tuple] = None,):
        """
        Inputs are expected to be memory mapped numpy arrays (.npy)

        Args:()
            images (np.ndarray): Memory mapped numpy array of images
            labels (np.ndarray): Memory mapped numpy array of labels
            transform (Optional[torch.nn.Module], optional): Torchvision transform to apply to images.
            validation_indices (Optional[np.ndarray], optional): Indices to use for validation set during cross validation.
        """
        self.images = images
        self.labels = labels
        self.indices = validation_indices

        # Standard mean and std values for ResNet
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Convert mean and std to tensors with shape [C, 1, 1]
        self.mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)


    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx) -> Tuple:
        image = self.images[idx]
        label = self.labels[idx]
        
        # Normalize the image
        image = torch.tensor(image, dtype=torch.float32)
        image.div_(255.0)
        if len(image.shape) == 4:
            image = (image - self.mean_tensor.unsqueeze(0)) / self.std_tensor.unsqueeze(0)
        else:
            image = (image - self.mean_tensor) / self.std_tensor
            
        return image, torch.tensor(label, dtype=torch.long)

    def split(self, split_ratio: float):
        split_index = int(self.images.shape[0] * split_ratio)
        # Create views for training and validation sets
        train_images = self.images[:split_index]
        val_images = self.images[split_index:]
        
        train_labels = self.labels[:split_index]
        val_labels = self.labels[split_index:]
        
        train_dataset = MemmapDataset(train_images, train_labels)
        val_dataset = MemmapDataset(val_images, val_labels)
        
        return train_dataset, val_dataset
    
    def split_into_folds(self, num_folds: int) -> List[Dataset]:
        """
        Creates a list of validation datasets for cross validation.
        The original dataset will be used as the training dataset.

        When training, make sure the indices from the validation dataset are not included in 
        the training batch.
        """
        fold_size = self.images.shape[0] // num_folds
        validation_datasets = []

        for i in range(num_folds):
            begin = i * fold_size
            end = (i + 1) * fold_size

            val_images = self.images[begin:end]
            val_labels = self.labels[begin:end]

            validation_datasets.append(MemmapDataset(val_images, val_labels, validation_indices=(begin, end)))
        
        return validation_datasets


def slice_collate_fn(batch):
    """
        Returns the slice as the batch.
    """
    return batch[0]

class SliceSampler(Sampler):
    """
    Takes slices of the dataset to minimize overhead of accessing a memory mapped array.
    Can optionally skip indices to allow for cross validation with memory mapping.
    """
    def __init__(self, dataset_len, batch_size, skip_indices: Optional[Tuple] = None):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.start_skip = None
        self.end_skip = None
        if skip_indices:
            self.start_skip = skip_indices[0]
            self.end_skip = skip_indices[1]
       

    def __iter__(self):
        for start_idx in range(0, self.dataset_len, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.dataset_len)
            
            if self.start_skip is None:
                yield slice(start_idx, end_idx)
                continue
            
            # Check for any indices we want to skip
            if start_idx >= self.start_skip and start_idx <= self.end_skip or end_idx >= self.start_skip and end_idx <= self.end_skip:
                continue  # Skip this slice
            
            yield slice(start_idx, end_idx)

    def __len__(self):
        return (self.dataset_len + self.batch_size - 1) // self.batch_size  # number of batches
    
class SliceBatchSampler(BatchSampler):
    """
    Passes along the batch untouched.
    """
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)
    def __iter__(self):
        for batch in super().__iter__():
            yield batch
    def __len__(self):
        return super().__len__()