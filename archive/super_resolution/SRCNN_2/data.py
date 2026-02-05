from PIL import Image
import numpy as np
import torch

class SRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, hr_images, scale_factor):
        self.hr_images = []
        self.lr_images = []
        for idx, hr_image in enumerate(hr_images):
            hr_image = Image.fromarray(hr_image)
            hr_image = hr_image.resize(((hr_image.width//scale_factor)*scale_factor, (hr_image.height//scale_factor)*scale_factor), resample=Image.BICUBIC)
            lr_image = hr_image.resize((hr_image.width//scale_factor, hr_image.height//scale_factor), resample=Image.BICUBIC)  
            lr_image = lr_image.resize((lr_image.width*scale_factor, lr_image.height*scale_factor), resample=Image.BICUBIC)

            self.hr_images.append(np.transpose(hr_image, (2, 0, 1)))
            self.lr_images.append(np.transpose(lr_image, (2, 0, 1)))
    def __len__(self):
        return len(self.hr_images)
    def __getitem__(self, index):
        hr_image = self.hr_images[index]
        lr_image = self.lr_images[index]

        hr_image = torch.tensor(hr_image, dtype=torch.float32)
        hr_image.div_(255)

        lr_image = torch.tensor(lr_image, dtype=torch.float32)
        lr_image.div_(255)

        return lr_image, hr_image