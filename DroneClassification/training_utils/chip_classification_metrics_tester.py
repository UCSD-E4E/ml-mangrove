import torch
from DroneClassification.training_utils.training_utils import ChipClassificationMetrics

# Simulate logits for 3 samples, 4 classes
pred_logits = torch.tensor([
    [2.0, 1.0, 0.5, 0.2],
    [0.1, 0.2, 0.3, 3.0],
    [0.5, 2.0, 1.0, 0.1]
])

# True labels
labels = torch.tensor([0, 3, 1])

acc = ChipClassificationMetrics.accuracy(pred_logits, labels)
print("Accuracy =", acc)
