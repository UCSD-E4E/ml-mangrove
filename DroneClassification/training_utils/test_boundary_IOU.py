import torch
import numpy as np
from DroneClassification.training_utils import boundary_iou

# Create two simple masks
gt = np.zeros((10, 10), dtype=np.uint8)
pred = np.zeros((10, 10), dtype=np.uint8)

# Add a square region
gt[2:7, 2:7] = 1
pred[3:8, 3:8] = 1   # shifted boundary

score = boundary_iou(gt, pred)
print("Boundary IoU =", score)
