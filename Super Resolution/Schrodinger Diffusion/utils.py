import numpy as np
import matplotlib.pyplot as plt
import torch
from rasterio.plot import show

class JupyterArgParser:
    def __init__(self):
        self.args = {}

    def add_argument(self, name, type=None, default=None, help=None, action=None):
        self.args[name.lstrip('--').replace('-', '_')] = {
            'type': type,
            'value': default,
            'action': action,
        }

    def set_value(self, name, value):
        if name in self.args:
            self.args[name]['value'] = value
        else:
            raise KeyError(f"Argument '{name}' not found.")
        
    def get_options(self):
        opts = Opt()
        for name, config in self.args.items():
            value = config['value'] if config['value'] is not None else False
            setattr(opts, name, value)

        return opts

class Opt:
    """
    A simple class to hold parsed options.
    """
    def __init__(self):
        pass


# Helper Functions

# returns dictionary of {pixel_value: counts}
def pixel_value_counts(arr):
    values, counts = np.unique(arr, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)} 

# takes in list of satellite tile pathnames
# calculates per-band mean and standard deviation
def calculate_band_stats(path_names, batch_size=32):
    print("Calculating per-band stats:")

    running_sums = np.zeros(13, dtype=np.float64)
    running_sq_sums = np.zeros(13, dtype=np.float64)
    total_pixels = 0

    for path in path_names:
        satellite_tiles = np.load(path, mmap_mode="r")
        sB, sC, sH, sW = satellite_tiles.shape
        assert sC == 12, f"Expected 12 bands, got {sC}"

        num_batches = int(np.ceil(sB / batch_size))

        for b in range(num_batches):
            if (b % 200 == 0):
                print(f"processing batch {b}/{num_batches}")

            batch = satellite_tiles[b * batch_size: (b+1) * batch_size]
            zero_channel = np.zeros((batch.shape[0], 1, sH, sW), dtype=batch.dtype)
            batch = np.concatenate((batch[:, :10], zero_channel, batch[:, 10:]), axis=1)
            batch = batch.astype(np.float32) / 255.0

            for band in range(13):
                band_pixels = batch[:, band, :, :].reshape(-1)
                running_sums[band] += band_pixels.sum()
                running_sq_sums[band] += np.square(band_pixels).sum()

            total_pixels += batch.shape[0] * sH * sW
    
    means = running_sums / total_pixels
    variances = (running_sq_sums / total_pixels) - (means ** 2)
    stds = np.sqrt(variances)

    return means.tolist(), stds.tolist()

# takes in img_tensor, mean_tensor, std_tensor and returns normalized iamge
def normalize_image(img_tensor, mean_tensor, std_tensor):
    # print("running normalize_image")
    # check that img_tensor, mean_tensor, and std_tensor have same number of channels
    assert img_tensor.shape[0] == mean_tensor.shape[0] == std_tensor.shape[0], f"tensors do not have the same channel number: {img_tensor.shape[0]}, {mean_tensor.shape[0]}, and {std_tensor.shape[0]}"
    assert img_tensor.shape[0] == 3 or img_tensor.shape[0] == 13, "img_tensor does not have either 3 or 13 channels"

    img_tensor = (img_tensor - mean_tensor) / std_tensor

    return img_tensor

def denormalize_image(normalized_tensor, mean_tensor, std_tensor):
    # print("running denormalize_image")
    # check that img_tensor, mean_tensor, and std_tensor have same number of channels
    assert normalized_tensor.shape[0] == mean_tensor.shape[0] == std_tensor.shape[0], f"tensors do not have the same channel number: {normalized_tensor.shape[0]}, {mean_tensor.shape[0]}, and {std_tensor.shape[0]}"
    assert normalized_tensor.shape[0] == 3 or normalized_tensor.shape[0] == 13, "normalized_tensor does not have either 3 or 13 channels"

    img_tensor = normalized_tensor * std_tensor + mean_tensor

    return img_tensor

# plot distribution of given latent space. sample 200 random latent spaces from the population
def plot_latent_distribution(latent_arr, num_images=200):
    assert latent_arr.ndim == 4, "Expected latent array of shape (B, C, H, W)"
    latent_arr_B = latent_arr.shape[0]

    random_indices = np.random.choice(latent_arr_B, size=num_images, replace=False)

    sampled_values = []
    for idx in random_indices:
        latent = latent_arr[idx]
        sampled_values.append(latent.flatten())

    all_values = np.concatenate(sampled_values)

    plt.hist(all_values, bins=100, density=True, alpha=0.6, color='g')
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()

# plot distribution of one latent space
def plot_one_latent_distribution(latent_arr):
    assert latent_arr.ndim == 3, "Expected latent array of shape (C, H, W)"

    all_values = latent_arr.flatten()

    plt.hist(all_values, bins=100, density=True, alpha=0.6, color='g')
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)
    plt.show()

def plot_img_gt_pred(img_tensor, gt_tensor, pred_tensor, rgb_mean_tensor, rgb_std_tensor, satellite_mean_tensor, satellite_std_tensor):
    SAT_RGB_CHANNELS = [3, 2, 1]

    fig, ax = plt.subplots(figsize=(3, 3))
    if (img_tensor.shape[0] == 13):
        img_tensor = denormalize_image(img_tensor, satellite_mean_tensor, satellite_std_tensor)
        img_tensor = img_tensor * 255.0
        img_tensor = torch.clamp(img_tensor, 0, 255).to(torch.uint8)
        img_arr = img_tensor[SAT_RGB_CHANNELS, :, :].cpu().numpy().astype(np.uint8)
        show(img_arr, ax=ax)
    elif (img_tensor.shape[0] == 3):
        img_tensor = denormalize_image(img_tensor, rgb_mean_tensor, rgb_std_tensor)
        img_tensor = img_tensor * 255.0
        img_tensor = torch.clamp(img_tensor, 0, 255).to(torch.uint8)
        img_arr = img_tensor.cpu().numpy().astype(np.uint8)
        show(img_arr, ax=ax)
    else:
        print(f"img_tensor has {img_tensor.shape[0]} bands. invalid input")
        return

    # show ground truth label
    plt.figure(figsize=(3, 3))
    gt_arr = gt_tensor.cpu().numpy().squeeze()
    plt.imshow(gt_arr, cmap='viridis', vmin=0, vmax=1)
    plt.show()

    # show pred label probabiltiies
    plt.figure(figsize=(3, 3))
    pred_probabilities = pred_tensor.sigmoid().cpu().numpy().squeeze()
    plt.imshow(pred_probabilities, cmap='viridis', vmin=0, vmax=1)
    plt.show()

def custom_mse_loss(pred_tensor, true_tensor):
    diff = pred_tensor - true_tensor
    squared_diff = diff ** 2
    mse = squared_diff.mean()
    return mse


def plot_latent_space(latent_tensor):
  latent_arr = latent_tensor.cpu().numpy()

  fig, axes = plt.subplots(16, 16, figsize=(12, 12))
  for i, ax in enumerate(axes.flat):
    ax.imshow(latent_arr[i], cmap='viridis')
    ax.axis('off')
  plt.tight_layout()
  plt.show()


# plot distribution of one latent space
def plot_band_distribution(satellite_arr, num_images=200):
    assert satellite_arr.ndim == 4, "Expected latent array of shape (B, C, H, W)"

    B, C, H, W = satellite_arr.shape

    random_indices = np.random.choice(B, size=num_images, replace=False)

    plt.figure(figsize=(10, 6))
    for c in range(C):
        channel_values = satellite_arr[random_indices, c, :, :].reshape(-1)
        plt.hist(channel_values, bins=100, density=True, alpha=0.6, label=f'Channel {c}', color='g')
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.grid(True)
        plt.show()
