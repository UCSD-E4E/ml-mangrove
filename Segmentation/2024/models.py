import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Conv2d, Module
from torchgeo.models import resnet18, resnet50, get_weight
from torchvision.models.resnet import ResNet
from typing import Optional
from satlaspretrain_models import Weights
weights_manager = Weights()

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
            )
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
    def __init__(self, ResNet18 : Optional[ResNet] = None, input_image_size=128):
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
            )
        )

    def forward(self, image):
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
    
class SwinB_UNet(Module):
    """
    Encoder: Swin Transformer pretrained on Satlas 0.5-2 m/pixel aerial images
    Decoder: 

    Backbone model outputs a feature pyramid network with 5 levels, with the first level being the dimensions 
    of the input image and the last level being 1/32 of the input image dimensions.

    512px:  512 -> 128 -> 64 -> 32 -> 16
    256px: 256 -> 64 -> 32 -> 16 -> 8
    128px: 128 -> 32 -> 16 -> 8 -> 4
    """
    def __init__(self, image_size=128, num_classes=1):
        super().__init__()
        # Load the pretrained model to cpu
        self.model = weights_manager.get_pretrained_model(
            "Aerial_SwinB_SI", fpn=True, device='cpu'
        )
        self.image_size = image_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(image_size, image_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(image_size, image_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(image_size, image_size, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(image_size, image_size, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(image_size, image_size, kernel_size=3, padding=1)
        self.classification_head = nn.Conv2d(image_size, num_classes, kernel_size=1)
        
    def forward(self, image):
        image = image[:, :3, :, :]
        fpn_outputs = self.model(image)

        # Upsample each FPN output to the size of the first tensor (128x128)
        upsampled_features = [
            self.conv1(fpn_outputs[0]),
            F.interpolate(self.conv2(fpn_outputs[1]), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False),
            F.interpolate(self.conv3(fpn_outputs[2]), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False),
            F.interpolate(self.conv4(fpn_outputs[3]), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False),
            F.interpolate(self.conv5(fpn_outputs[4]), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False),
        ]

        # Combine upsampled features
        x = sum(upsampled_features)
        
        return self.classification_head(x)


"""
  _____    _    __    __                 _                 
 |  __ \  (_)  / _|  / _|               (_)                
 | |  | |  _  | |_  | |_   _   _   ___   _    ___    _ __  
 | |  | | | | |  _| |  _| | | | | / __| | |  / _ \  | '_ \ 
 | |__| | | | | |   | |   | |_| | \__ \ | | | (_) | | | | |
 |_____/  |_| |_|   |_|    \__,_| |___/ |_|  \___/  |_| |_|
                                                                                                                  
"""
class MangroveDiffusion(Module):
    def __init__(self, decoder : Module, encoder : Module,
                  feature_size: int = 2048, diffusion_weights = None):
        super().__init__()
        
        self.encoder = encoder
        self.encoder.requires_grad_(False)

        self.adaptive_pool = torch.nn.AdaptiveMaxPool2d((9,9))
        
        self.decoder = decoder
        self.decoder.requires_grad_(False)
        
        self.reverse_diffuser = torch.nn.Linear(feature_size + feature_size + 1, feature_size)
    
    def forward(self, image):
        x = self.encoder(image)
        x = self.adaptive_pool(x)
        x = self.decoder(x)
        return x

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
        inputs = torch.sigmoid(inputs)
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


class TverskyLoss(nn.Module):
    """
    The Tversky index is a generalization of the Dice coefficient and Jaccard index.
    It allows control over the trade-off between false positives and false negatives
    through the parameters alpha and beta.

    Parameters:
    -----------
    alpha : float, optional (default=0.5)
        The weight of false positives.
    beta : float, optional (default=0.5)
        The weight of false negatives.
    smooth : float, optional (default=1e-5)
        A smoothing factor to avoid division by zero errors.
    """

    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
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
        return 1 - Tversky

class CustomLoss(nn.Module):
    def __init__(self, beta=0.9, smooth=1e-5, pos_weight=None):
        super(CustomLoss, self).__init__()
        self.beta = beta
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        fn_loss = self.beta * FN / (TP + self.smooth)
        return bce_loss + fn_loss

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

class FPNUpsample(Module):
    """
    Helper class for the SwinB UNet architecture.
    Uses convolutional layers to upsample the input.
    """
    def __init__(self, in_channels, out_channels, in_size):
        super(FPNUpsample, self).__init__()
        self.upsampler = nn.Sequential(
            nn.Conv2d(in_size, in_size, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(
              size=(256, 256),
              mode="bilinear",
              align_corners=False,
            )
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