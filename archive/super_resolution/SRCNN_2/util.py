import math
import torch
import numpy as np

def psnr(label, pred):
    label = label.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    diff = pred - label
    rmse = math.sqrt(np.mean((diff) ** 2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(1 / rmse)
        return PSNR
