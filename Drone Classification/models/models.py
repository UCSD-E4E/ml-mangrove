import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.nn import Conv2d, Module
from torchgeo.models import resnet18, resnet50, get_weight
from torchvision.models.resnet import ResNet
from typing import Optional

"""
Pretrained model Weights from SSL4EO-12 dataset
@ https://github.com/zhu-xlab/SSL4EO-S12

Imported using torchgeo
@ https://torchgeo.readthedocs.io/en/stable/api/models.html

SWIN high resolution aerial imagery model from Satlas
@ https://github.com/allenai/satlas

New weights can be imported from torchgeo using:
torchgeo.models.get_weight("ResNet50_Weights.SENTINEL2_ALL_MOCO")
"""

"""
   _____   _                       _    __   _                     
  / ____| | |                     (_)  / _| (_)                    
 | |      | |   __ _   ___   ___   _  | |_   _    ___   _ __   ___ 
 | |      | |  / _` | / __| / __| | | |  _| | |  / _ \ | '__| / __|
 | |____  | | | (_| | \__ \ \__ \ | | | |   | | |  __/ | |    \__ \
  \_____| |_|  \__,_| |___/ |___/ |_| |_|   |_|  \___| |_|    |___/
                                                                   
"""                                                             

class ResNet50_UNet(Module):
    """
    UNet architecture with ResNet50 encoder. 
    
    Default ResNet is trained on Sentinel-2 3 channel RGB satellite imagery.
    ResNet50 models trained on other inputs can be used.
    
    """
    def __init__(self, ResNet50 : Optional[ResNet] = None, num_channels=3, input_image_size=256):
        super().__init__()
        self.num_channels = num_channels
        
        if ResNet50 == None:
            ResNet50=resnet50(
                weights=get_weight("ResNet50_Weights.SENTINEL2_RGB_SECO")
            )
        
        # Pretrained Encoder with frozen weights
        for param in ResNet50.parameters():
            param.requires_grad = False
        self.layer1 = nn.Sequential(
            ResNet50.conv1,
            ResNet50.bn1,
            nn.ReLU(),
            ResNet50.maxpool,
            ResNet50.layer1,
        )
        self.layer2 = ResNet50.layer2
        self.layer3 = ResNet50.layer3
        self.layer4 = ResNet50.layer4

        # Center
        self.center = Upsample(2048, 1536, 1024)

        # Skip connections
        self.skip_conv1 = Conv2d(1024, 1024, kernel_size=1)
        self.skip_conv2 = Conv2d(512, 512, kernel_size=1)
        self.skip_conv3 = Conv2d(256, 256, kernel_size=1)

        # Decoder
        self.decoder1 = Decoder(1024+1024, 1024, 512)
        self.decoder2 = Decoder(512+512, 512, 256)
        self.classification_head = nn.Sequential(
            Decoder(256+256, 256, 128),
            Upsample(128, 128, 64),
            Conv2d(64, 1, kernel_size=1),
            nn.Upsample(
              size=(input_image_size, input_image_size),
              mode="bilinear",
              align_corners=False,
            ),
        )
    
    def forward(self, image):
        image = image[:, :self.num_channels, :, :]
        
        # Encode
        x1 = self.layer1(image)  # 256
        x2 = self.layer2(x1)  # 512
        x3 = self.layer3(x2)  # 1024
        x4 = self.layer4(x3)  # 2048

        # Center
        x = self.center(x4)
        
        # decode
        x = torch.cat((x, self.skip_conv1(x3)), dim=1)
        x = self.decoder1(x)
        x = torch.cat((x, self.skip_conv2(x2)), dim=1)
        x = self.decoder2(x)
        x = torch.cat((x, self.skip_conv3(x1)), dim=1)
        x = self.classification_head(x)

        return x

class ResNet18_UNet(Module):
    """
    UNet architecture with ResNet18 encoder.
    
    Default ResNet is trained on Sentinel-2 3 channel RGB satellite imagery.
    """
    def __init__(self, ResNet18 : Optional[ResNet] = None, input_image_size=256):
        super(ResNet18_UNet, self).__init__()
        if ResNet18 is None:
            ResNet18 = resnet18(
                weights=get_weight("ResNet18_Weights.SENTINEL2_RGB_SECO")
            )
        
        for param in ResNet18.parameters():
            param.requires_grad = False
        
        self.layer1 = nn.Sequential(
            ResNet18.conv1,
            ResNet18.bn1,
            nn.ReLU(),
            ResNet18.maxpool,
            ResNet18.layer1,
        )
        self.layer2 = ResNet18.layer2
        self.layer3 = ResNet18.layer3
        self.layer4 = ResNet18.layer4

        # Center
        self.center = Decoder(512, 312, 256)

        # Skip connections
        self.skip_conv1 = Conv2d(256, 256, kernel_size=1)
        self.skip_conv2 = Conv2d(128, 128, kernel_size=1)
        self.skip_conv3 = Conv2d(64, 64, kernel_size=1)

        #decoder
        self.decoder1 = Decoder(256+256, 256, 128)
        self.decoder2 = Decoder(128+128, 128, 64)
        
        self.classification_head = nn.Sequential(
            Upsample(64+64, 64, 32),
            Conv2d(32, 1, kernel_size=2, padding=1),
            nn.Upsample(
              size=(input_image_size, input_image_size),
              mode="bilinear",
              align_corners=False,
            ),
        )

    def forward(self, image):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image[:, :3, :, :]

        # Encode
        x1 = self.layer1(image)  # 64
        x2 = self.layer2(x1)  # 128
        x3 = self.layer3(x2)  # 256
        x4 = self.layer4(x3)  # 512
      
        # Center
        x = self.center(x4)

        # Decode
        x = torch.cat((x, self.skip_conv1(x3)), dim=1)
        x = self.decoder1(x)
        x = torch.cat((x, self.skip_conv2(x2)), dim=1)
        x = self.decoder2(x)
        x = torch.cat((x, self.skip_conv3(x1)), dim=1)
        x = self.classification_head(x)

        return x

class ResNet_FC(Module):
    """
    ResNet with Fully Connected output layer.
    """
    def __init__(self, ResNet : Optional[ResNet] = None, num_classes=1, input_image_size=128):
        super(ResNet_FC, self).__init__()
        self.num_classes = num_classes
        self.input_image_size = input_image_size
        if ResNet is None:
            ResNet = resnet18(
                weights=get_weight("ResNet18_Weights.SENTINEL2_RGB_SECO")
            )
        ResNet.fc = nn.Identity()
        self.resnet = ResNet

        dummy_input = torch.randn(1, 3, input_image_size, input_image_size)
        features = ResNet(dummy_input)
        feature_dim = features.shape[1]
        # Add a fully connected layer for binary segmentation
        self.classification_head = nn.Linear(feature_dim, input_image_size*input_image_size * num_classes)

    def forward(self, image):
        image = image[:, :3, :, :]
        x = self.resnet(image)
        x = x.flatten(start_dim=1)
        x = self.classification_head(x)
        x = x.view(-1, self.num_classes, self.input_image_size, self.input_image_size)
        return x

class SegmentModelWrapper(Module):
    def __init__(self, model: nn.Module, threshold=0.5):
        super(SegmentModelWrapper, self).__init__()
        self.model = model
        self.model.eval()
        self.threshold = threshold
        
        # Standard mean and std values for ResNet
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Convert mean and std to tensors with shape [C, 1, 1]
        self.mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    
    def forward(self, image: np.ndarray):
        """
        Expects a numpy array of dimensions CxHxW.
         
        It can also accept batched images of size BxCxHxW.
        """
        image = torch.tensor(image, dtype=torch.float32)
        image.div_(255.0)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        image = (image - self.mean_tensor) / self.std_tensor
        
        out = self.model.forward(image)

        out = torch.sigmoid(out)

        return (out > self.threshold).to(torch.uint8)


"""
  _      ____   _____ _____ 
 | |    / __ \ / ____/ ____|
 | |   | |  | | (___| (___  
 | |   | |  | |\___ \\___ \ 
 | |___| |__| |____) |___) |
 |______\____/|_____/_____/ 
                           
"""
class BCEDiceLoss(nn.Module):
    def __init__(self, weight: Optional[torch.tensor] = None, size_average=True):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=weight, reduction='mean' if size_average else 'sum')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + 1) / (inputs.sum() + targets.sum() + 1)
        dice_loss = 1 - dice
        return bce_loss + dice_loss
    
class BCETverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-5, pos_weight=None):
        super(BCETverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, inputs, targets):
        # BCE Loss
        bce_loss = self.bce(inputs, targets)
        
        # Apply sigmoid to the inputs
        inputs = torch.sigmoid(inputs)
        
        # Flatten the tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # True positives, false positives, and false negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        
        # Tversky index
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        tversky_loss = 1 - Tversky
        
        return bce_loss + tversky_loss

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

class FocalLoss(nn.Module):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    def __init__(self, alpha=0.1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{self.reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss

"""
  _    _          _                             
 | |  | |        | |                            
 | |__| |   ___  | |  _ __     ___   _ __   ___ 
 |  __  |  / _ \ | | | '_ \   / _ \ | '__| / __|
 | |  | | |  __/ | | | |_) | |  __/ | |    \__ \
 |_|  |_|  \___| |_| | .__/   \___| |_|    |___/
                     | |                        
                     |_|                        
"""
class Upsample(Module):
    """
    Helper class for the UNet architecture.
    Uses convolutional layers to upsample the input.
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Upsample, self).__init__()
        self.upsampler = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )
    
    def forward(self, x) -> torch.Tensor:
        return self.upsampler(x)

class Decoder(Module):
    """
    Helper class for the UNet architecture.
    Uses convolutional layers to upsample the input.
    Includes dropout layer to prevent overfitting.
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x) -> torch.Tensor:
        return self.decoder(x)

class Downsample(Module):
    """
    Helper class for the UNet architecture.
    Uses convolutional layers to downsample the input.

    Increases channels by a factor of 2 and reduces the spatial dimensions by half.
    """
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.downsampler = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x) -> torch.Tensor:
        return self.downsampler(x)