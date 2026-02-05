from torch.utils.data import Dataset
import numpy as np
from typing import Optional, List, Tuple
import torch

class SuperResolutionDataset(Dataset):
    def __init__(self, hr_images, lr_images, transform=None):
        self.hr_images = hr_images
        self.lr_images = lr_images
        self.transform = transform

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_image = self.hr_images[idx]
        lr_image = self.lr_images[idx]
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
        return lr_image, hr_image

# Example usage
# hr_images and lr_images should be lists of image paths or preloaded image arrays
# hr_images = [...]  # Your high-resolution images
# lr_images = [...]  # Your low-resolution images
# dataset = SuperResolutionDataset(hr_images, lr_images, transform=ToTensor())
# data_loader = DataLoader(dataset, batch_size=16, shuffle=True)


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