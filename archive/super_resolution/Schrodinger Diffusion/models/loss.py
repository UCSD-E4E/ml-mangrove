import torch
import torch.nn as nn

class LandmassLoss(nn.Module):
    """
    A Loss function to encourage the model to accurately predict the overall mangrove coverage in the image.

    This will allow the model to train on the overall size of the mangrove area, rather than individual pixels.
    It is also a very simple calculation - essentially the normalized absolute difference.
    """
    def __init__(self):
        super(LandmassLoss, self).__init__()
    def forward(self, prediction: torch.tensor, target: torch.tensor):
        return (prediction.sum() - target.sum()) / (target.sum() + 1)

class JaccardLoss(nn.Module):
    def __init__(self, smooth=1e-10):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred : torch.tensor, y_true: torch.tensor):
        y_pred = torch.sigmoid(y_pred)
        
        # Flatten the tensors to simplify the calculation
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        # Calculate intersection and union
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum() - intersection
        
        # Calculate the Jaccard index
        jaccard_index = (intersection + self.smooth) / (union + self.smooth)
        
        # Return the Jaccard loss (1 - Jaccard index)
        return 1 - jaccard_index