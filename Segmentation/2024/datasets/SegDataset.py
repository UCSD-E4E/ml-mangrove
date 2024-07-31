from torch.utils.data import Dataset
import numpy as np
import torch


class SegmentationDataset(Dataset):
    def __init__(self, images, labels, transforms=None):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.transforms = transforms
        self.images = np.array(images)
        self.labels = np.array(labels)

    def __len__(self):
        # return the number of total samples contained in the dataset
        return np.array(self.images).shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # load the image from disk, swap its channels from BGR to RGB,

        # and read the associated mask from disk in grayscale mode

        # Apply transformations to image and label
        if self.transforms is not None:
            image = self.transforms(image)
            label = self.transforms(label)
            
        # return a tuple of the image and its mask
        return (image, torch.Tensor(label))