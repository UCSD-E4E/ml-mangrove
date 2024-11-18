from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

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
hr_images = [...]  # Your high-resolution images
lr_images = [...]  # Your low-resolution images
dataset = SuperResolutionDataset(hr_images, lr_images, transform=ToTensor())
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
