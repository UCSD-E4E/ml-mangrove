import os
import numpy as np
import torch
from einops import rearrange
from typing import Tuple, Optional
from tqdm import tqdm


def multispectral_normalization(images, target_superres: int = 10, batch_size: Optional[int] = None):
    """
    Normalize a Sentinel-2 batch tensor to [0,1] using per-band percentile scaling.
    
    Ideally, the percentiles should be computed on a large sample of the dataset to be robust.
    If the dataset is large, calculate normalization metrics in batches from a memory-mapped file. Specifying `batch_size` will enable this behavior.

    Normalization method based on implementation from:
    https://medium.com/sentinel-hub/how-to-normalize-satellite-images-for-deep-learning-d5b668c885af

    Args:
        images: shape (B, C, H, W)
        target_superres (int): the target super-resolution factor for bilinear interpolation.
        batch_size (Optional[int]): size of each batch for processing, if None, process the entire dataset at once.
    Returns:
        images: normalized images, shape (B, C, H * target_superres, W * target_superres), dtype uint8
    """
    B, C, H, W = images.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


    if batch_size is None:
        
        # Convert to tensor if not already
        if not isinstance(images, torch.Tensor):
            images = _arrange_and_interpolate(torch.from_numpy(images).to(device), target_superres)
        else:
            images = _arrange_and_interpolate(images.to(device), target_superres)

        # If the input is a 3D tensor, add a batch dimension
        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        # Normalize the images
        lower, upper = _find_normalization_percentiles(images, lower=1, upper=99, sample_frac=0.01)
        images = _normalize_by_percentile(images, lower, upper)

        # quantize
        # SSL4EO-12 found no performance difference between uint8 and float32 precision
        # src: https://github.com/zhu-xlab/SSL4EO-S12
        images = (images * 255).to(torch.uint8)

    else:  # Process in batches using memory-mapping
        
        new_images = np.lib.format.open_memmap('normalized_multispectral_images.npy', dtype='float32', mode='w+', shape=(B, C, H * target_superres, W * target_superres))
        lower_percentiles, upper_percentiles = [], []

        # Find normalization percentiles for the entire dataset
        for i in tqdm(range(0, B, batch_size), desc="Calculating normalization statistics...", total=(B + batch_size - 1) // batch_size):

            # Pull tensor from memory-mapped array
            images_batch = _arrange_and_interpolate(torch.from_numpy(images[i:i + batch_size]).to(device), target_superres)

            # Find percentiles for the current batch
            p1, p99 = _find_normalization_percentiles(images_batch, lower=1, upper=99, sample_frac=0.01)
            lower_percentiles.append(p1)
            upper_percentiles.append(p99)

            new_images[i:i + batch_size] = images_batch.cpu().numpy()
            del images_batch  # Free memory

            # Ensure data is written to disk
            new_images.flush()  
            images.flush()
        
        if B % batch_size != 0:
            images_batch = _arrange_and_interpolate(torch.from_numpy(images[B - B % batch_size:]).to(device), target_superres)

            p1, p99 = _find_normalization_percentiles(images_batch, lower=1, upper=99, sample_frac=0.01)
            lower_percentiles.append(p1)
            upper_percentiles.append(p99)

            new_images[B - B % batch_size:] = images_batch.cpu().numpy()
            del images_batch
            new_images.flush()
            images.flush()
        del images

        # Compute the median percentiles across all batches
        lower_percentile = torch.median(torch.stack(lower_percentiles), dim=0).values
        upper_percentile = torch.median(torch.stack(upper_percentiles), dim=0).values
        del lower_percentiles, upper_percentiles

        # Save the normalized images as uint8
        images = np.lib.format.open_memmap('normalized_multispectral_images_uint8.npy', dtype='uint8', mode='w+', shape=new_images.shape)

        # Normalize the dataset using calculated percentiles
        for i in tqdm(range(0, B, batch_size), desc="Normalizing batches...", total=(B + batch_size - 1) // batch_size):
            batch = torch.from_numpy(new_images[i:i + batch_size]).to(device)
            # Normalize the batch
            batch = _normalize_by_percentile(batch, lower_percentile, upper_percentile)
            images[i:i + batch_size] = (batch.cpu().numpy() * 255).astype(np.uint8)  # Convert to uint8

            # Memory management
            del batch
            new_images.flush()
            images.flush()

        if B % batch_size != 0:
            batch = torch.from_numpy(new_images[B - B % batch_size:]).to(device)
            batch = _normalize_by_percentile(batch, lower_percentile, upper_percentile)
            images[B - B % batch_size:] = (batch.cpu().numpy() * 255).astype(np.uint8)  # Convert to uint8
            del batch
            images.flush()

        del new_images
        os.remove('normalized_multispectral_images.npy')  # Remove the original float32 memory-mapped file

    return images


def _find_normalization_percentiles(batch: torch.Tensor, lower=1, upper=99, sample_frac=0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the lower and upper percentiles of a batch tensor for normalization.

    Args:
        batch (Tensor): shape (B, C, H, W), float tensor on GPU
        lower (int): lower percentile (e.g., 1)
        upper (int): upper percentile (e.g., 99)
        sample_frac (float): fraction of pixels to sample for percentile estimation

    Returns:
       percentiles (tuple[Tensor, Tensor]): lower percentile, upper percentile

    """
    B, C, H, W = batch.shape
    N = int(B * H * W * sample_frac)

    flat = batch.permute(1, 0, 2, 3).reshape(C, -1)
    idx = torch.randint(0, flat.size(1), (N,), device=batch.device)
    sampled = flat[:, idx]  # (C, N)

    p1 = torch.quantile(sampled, lower / 100.0, dim=1)
    p99 = torch.quantile(sampled, upper / 100.0, dim=1)

    return p1, p99

def _normalize_by_percentile(image: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    Channels = image.shape[1]

    lower = lower.view(1, Channels, 1, 1)
    upper = upper.view(1, Channels, 1, 1)

    image = (image - lower) / (upper - lower)
    image = torch.clamp(image, 0.0, 1.0)

    return image

def _arrange_and_interpolate(images: torch.Tensor, target_superres: int = 10) -> torch.Tensor:
    _, C, H, W = images.shape
    
    if C > H or C > W:
        # Rearrange the channels to the second position if needed
        images = rearrange(images, 'b h w c -> b c h w')
        H, W = images.shape[2], images.shape[3]
    
    images = images.float()
    
    images = torch.nn.functional.interpolate(images, size=(H * target_superres, W * target_superres), mode='bilinear', align_corners=False)
    return images