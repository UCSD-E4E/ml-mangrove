from torch.utils.data import Dataset, Sampler, BatchSampler, DataLoader
from utils import *
import numpy as np
from typing import Optional, List, Tuple
from torchvision.transforms import ToTensor
import torch

# returns (normalized drone/sat image:label) pairs
class MemmapDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, validation_indices: Optional[Tuple] = None, rgb_means = None, rgb_sds = None, satellite_means = None, satellite_sds = None):
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

        self.rgb_means = rgb_means
        self.rgb_sds = rgb_sds
        self.satellite_means = satellite_means
        self.satellite_sds = satellite_sds

        # Standard mean and std values for ResNet
        if self.rgb_means == None:
            self.rgb_means = [0.485, 0.456, 0.406]
        if self.rgb_sds == None:
            self.rgb_sds = [0.229, 0.224, 0.225]
        # Check if satellite means are passed in
        if self.satellite_means == None or self.satellite_sds == None:
            print("missing satellite means or sds")
            return
        
        # Check that mean and sd arrays are the proper length
        assert len(self.rgb_means) == len(self.rgb_sds) == 3, "rgb_means or rgb_sds is not length 3"
        assert len(self.satellite_means) == len(self.satellite_sds) == 13, "satellite_means or satellite_sds is not length 13"

        # Convert mean and std to tensors with shape [C, 1, 1]
        self.rgb_mean_tensor = torch.tensor(self.rgb_means, dtype=torch.float32).view(3, 1, 1)
        self.rgb_std_tensor = torch.tensor(self.rgb_sds, dtype=torch.float32).view(3, 1, 1)
        self.satellite_mean_tensor = torch.tensor(self.satellite_means, dtype=torch.float32).view(13, 1, 1)
        self.satellite_std_tensor = torch.tensor(self.satellite_sds, dtype=torch.float32).view(13, 1, 1)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx) -> Tuple:
        image = self.images[idx]
        label = self.labels[idx]
        
        # Normalize the image
        image = torch.tensor(image, dtype=torch.float32)
        image.div_(255.0)
        
        if image.shape[0] == 3:
            image = normalize_image(image, self.rgb_mean_tensor, self.rgb_std_tensor)
        elif image.shape[0] == 13:
            image = normalize_image(image, self.satellite_mean_tensor, self.satellite_std_tensor)
        else:
            print("Error: image does not have either 3 or 13 channels")
            return
            
        return image, torch.tensor(label, dtype=torch.long)

    def split(self, split_ratio: float):
        split_index = int(self.images.shape[0] * split_ratio)
        # Create views for training and validation sets
        train_images = self.images[:split_index]
        val_images = self.images[split_index:]
        
        train_labels = self.labels[:split_index]
        val_labels = self.labels[split_index:]
        
        train_dataset = MemmapDataset(train_images, train_labels, rgb_means = self.rgb_means, rgb_sds = self.rgb_sds, satellite_means = self.satellite_means, satellite_sds = self.satellite_sds)
        val_dataset = MemmapDataset(val_images, val_labels, rgb_means = self.rgb_means, rgb_sds = self.rgb_sds, satellite_means = self.satellite_means, satellite_sds = self.satellite_sds)
        
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

            validation_datasets.append(MemmapDataset(val_images, val_labels, validation_indices=(begin, end)), rgb_means = self.rgb_means, rgb_sds = self.rgb_sds, satellite_means = self.satellite_means, satellite_sds = self.satellite_sds)
        
        return validation_datasets
    
class LatentSpaceDataset(Dataset):
    def __init__(self, hr_latents, lr_latents, transform=None):
        self.hr_latents = hr_latents
        self.lr_latents = lr_latents
        self.transform = transform

    def __len__(self):
        return len(self.hr_latents)

    def __getitem__(self, idx):
        hr_latent = self.hr_latents[idx]
        lr_latent = self.lr_latents[idx]
        if self.transform:
            hr_latent = self.transform(hr_latent)
            lr_latent = self.transform(lr_latent)
        return lr_latent, hr_latent

# returns sat rgb normalized : om rgb normalized pairs
class I2I_RGB_Dataset(Dataset):
    def __init__(self, sat_om: np.ndarray, rgb_om: np.ndarray, rgb_means = None, rgb_sds = None, satellite_means = None, satellite_sds = None):
        self.sat_om = sat_om
        self.rgb_om = rgb_om

        self.rgb_means = rgb_means
        self.rgb_sds = rgb_sds
        self.satellite_means = satellite_means
        self.satellite_sds = satellite_sds

        # Standard mean and std values for ResNet
        if self.rgb_means == None:
            self.rgb_means = [0.485, 0.456, 0.406]
        if self.rgb_sds == None:
            self.rgb_sds = [0.229, 0.224, 0.225]
        # Check if satellite means are passed in
        if self.satellite_means == None or self.satellite_sds == None:
            print("missing satellite means or sds")
            return
        
        # Check that mean and sd arrays are the proper length
        assert len(self.rgb_means) == len(self.rgb_sds) == 3, "rgb_means or rgb_sds is not length 3"
        assert len(self.satellite_means) == len(self.satellite_sds) == 13, "satellite_means or satellite_sds is not length 13"

        # Convert mean and std to tensors with shape [C, 1, 1]
        self.rgb_mean_tensor = torch.tensor(self.rgb_means, dtype=torch.float32).view(3, 1, 1)
        self.rgb_std_tensor = torch.tensor(self.rgb_sds, dtype=torch.float32).view(3, 1, 1)
        self.satellite_mean_tensor = torch.tensor(self.satellite_means, dtype=torch.float32).view(13, 1, 1)
        self.satellite_std_tensor = torch.tensor(self.satellite_sds, dtype=torch.float32).view(13, 1, 1)

    def __len__(self) -> int:
        return self.sat_om.shape[0]

    def __getitem__(self, idx) -> Tuple:
        sat_om = self.sat_om[idx]
        rgb_om = self.rgb_om[idx]

        bands = [3, 2, 1]
        sat_rgb_om = sat_om[bands, :, :]
        satellite_mean_tensor_rgb = self.satellite_mean_tensor[bands, :, :]
        satellite_std_tensor_rgb = self.satellite_std_tensor[bands, :, :]
        print(sat_rgb_om.shape, satellite_mean_tensor_rgb.shape, satellite_std_tensor_rgb.shape)
        
        # normalize sat_rgb_om
        sat_rgb_om = torch.tensor(sat_rgb_om, dtype=torch.float32)
        sat_rgb_om.div_(255.0)
        sat_rgb_om = normalize_image(sat_rgb_om, satellite_mean_tensor_rgb, satellite_std_tensor_rgb)

        # normalize rgb_om
        rgb_om = torch.tensor(rgb_om, dtype=torch.float32)
        rgb_om.div_(255.0)
        rgb_om = normalize_image(sat_rgb_om, self.rgb_mean_tensor, self.rgb_std_tensor)

        return sat_rgb_om, rgb_om

    def split(self, split_ratio: float):
        split_index = int(self.sat_om.shape[0] * split_ratio)
        # Create views for training and validation sets
        train_sat_om = self.sat_om[:split_index]
        val_sat_om = self.sat_om[split_index:]
        
        train_rgb_om = self.rgb_om[:split_index]
        val_rgb_om = self.rgb_om[split_index:]
        
        train_dataset = MemmapDataset(train_sat_om, train_rgb_om, rgb_means = self.rgb_means, rgb_sds = self.rgb_sds, satellite_means = self.satellite_means, satellite_sds = self.satellite_sds)
        val_dataset = MemmapDataset(val_sat_om, val_rgb_om, rgb_means = self.rgb_means, rgb_sds = self.rgb_sds, satellite_means = self.satellite_means, satellite_sds = self.satellite_sds)
        
        return train_dataset, val_dataset
  