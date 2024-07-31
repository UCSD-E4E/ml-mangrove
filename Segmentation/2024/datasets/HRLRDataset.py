from torch.utils.data import Dataset

class HRLRDataset(Dataset):
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
        
        return hr_image, lr_image
