from torch.utils.data import Dataset
import numpy as np
import torch

class SegmentationDataset(Dataset):
  def __init__(self, images, labels, transforms):
    # store the image and mask filepaths, and augmentation
		# transforms
    self.transforms = transforms
    self.images = images
    self.labels = labels
  def __len__(self):
    # return the number of total samples contained in the dataset
    return np.array(self.images).shape[0]
  def __getitem__(self, idx):
    try:
      image = self.images[idx]
      mask = self.labels[idx]
    except:
      print ("no index at", idx)
    #image = self.images[idx]
		# load the image from disk, swap its channels from BGR to RGB,

		# and read the associated mask from disk in grayscale mode
		# check to see if we are applying any transformations
      if self.transforms is not None:
			# apply the transformations to both image and its mask
        image = self.transforms(image)
        mask = self.transforms(mask)
		# return a tuple of the image and its mask
    return (image, torch.Tensor(mask).long())

