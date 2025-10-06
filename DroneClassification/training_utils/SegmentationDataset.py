from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision import tv_tensors
import torchvision.transforms.v2 as T

class SegmentationDataset(Dataset):
    def __init__(self, images, labels, transforms: T.Compose, load_to_ram: bool = True):
        """
        Inputs are expected to be memory-mapped numpy arrays (.npy) or in-RAM numpy arrays.
        """

        if isinstance(images, str):
            self.images = np.load(images, 'r+')
            if load_to_ram:
                images_ram = np.empty(self.images.shape, dtype=self.images.dtype)
                np.copyto(images_ram, self.images)
                self.images = images_ram
        else:
            self.images = images

        if isinstance(labels, str):
            self.labels = np.load(labels, 'r+')
            if load_to_ram:
                labels_ram = np.empty(self.labels.shape, dtype=self.labels.dtype)
                np.copyto(labels_ram, self.labels)
                self.labels = labels_ram
        else:
            self.labels = labels
        
        self.transforms = transforms


        # Normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3,1,1)
        self.std_tensor = torch.tensor(std, dtype=torch.float32).view(3,1,1)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()
        label = self.labels[idx].copy()

        image = tv_tensors.Image(torch.from_numpy(image).float())
        label = tv_tensors.Mask(torch.from_numpy(label).long())
        image, label = self.transforms(image, label)

        return image, label

    def split(self, split_ratio: float, valid_transforms: T.Compose) -> tuple:
        n = self.__len__()
        split_n = int(n * split_ratio)
        train  = SegmentationDataset(self.images[:split_n], self.labels[:split_n], transforms=self.transforms)
        valid = SegmentationDataset(self.images[split_n:], self.labels[split_n:], transforms=valid_transforms)
        return (train, valid)