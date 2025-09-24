import torch
import torch.nn as nn
import numpy as np
from torch.nn import Conv2d, Module
from torchgeo.models import resnet18, get_weight
import torchvision
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import resnet18 as tv_resnet18
from torchvision.models.resnet import ResNet
from typing import Optional

"""
Pretrained model Weights from SSL4EO-12 dataset
@ https://github.com/zhu-xlab/SSL4EO-S12

Imported using torchgeo
@ https://torchgeo.readthedocs.io/en/stable/api/models.html

New weights can be imported from torchgeo using:
torchgeo.models.get_weight("ResNet50_Weights.SENTINEL2_ALL_MOCO")
"""

r"""
   _____   _                       _    __   _                     
  / ____| | |                     (_)  / _| (_)                    
 | |      | |   __ _   ___   ___   _  | |_   _    ___   _ __   ___ 
 | |      | |  / _` | / __| / __| | | |  _| | |  / _ \ | '__| / __|
 | |____  | | | (_| | \__ \ \__ \ | | | |   | | |  __/ | |    \__ \
  \_____| |_|  \__,_| |___/ |___/ |_| |_|   |_|  \___| |_|    |___/
                                                                   
"""                                                             
class ResNet_UNet(Module):
    """
    UNet architecture with ResNet encoder.

    Defaults to ResNet18 with ImageNet weights.
    """
    def __init__(self, ResNet: Optional[ResNet] = None, input_image_size=224, num_classes=1):
        super(ResNet_UNet, self).__init__()
        if ResNet is None:
            ResNet = tv_resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        for param in ResNet.parameters():
            param.requires_grad = False
        
        self.layer1 = nn.Sequential(
            ResNet.conv1,
            ResNet.bn1,
            nn.ReLU(),
            ResNet.maxpool,
            ResNet.layer1,
        )
        self.layer2 = ResNet.layer2
        self.layer3 = ResNet.layer3
        self.layer4 = ResNet.layer4

        dummy_input = torch.randn(1, 3, input_image_size, input_image_size)
        x = self.layer1(dummy_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Define feature dimensions
        feature_dim = x.shape[1]
        half_dim = feature_dim // 2
        quarter_dim = feature_dim // 4
        eighth_dim = feature_dim // 8
        sixteenth_dim = feature_dim // 16
        
        # Center
        self.center = Decoder(feature_dim, int(feature_dim // 1.5), half_dim)

        # Skip connections
        self.skip_conv1 = Conv2d(half_dim, half_dim, kernel_size=1)
        self.skip_conv2 = Conv2d(quarter_dim, quarter_dim, kernel_size=1)
        self.skip_conv3 = Conv2d(eighth_dim, eighth_dim, kernel_size=1)

        #decoder
        self.decoder1 = Decoder(feature_dim, half_dim, quarter_dim)
        self.decoder2 = Decoder(half_dim, quarter_dim, eighth_dim)
        
        self.classification_head = nn.Sequential(
            Upsample(quarter_dim, eighth_dim, sixteenth_dim),
            Conv2d(sixteenth_dim, num_classes, kernel_size=2, padding=1),
            nn.Upsample(
              size=(input_image_size, input_image_size),
              mode="bilinear",
              align_corners=False,
            ),
            Conv2d(num_classes, num_classes, kernel_size=3, padding=1) # smooth output
        )

    def forward(self, image):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image[:, :3, :, :]

        # Encode
        x1 = self.layer1(image)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
      
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


class ResNet_UNet_NoSkip(Module):
    """
    ResNet UNet without any skip connections.
    """
    def __init__(self, ResNet : ResNet = tv_resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1), num_classes=1, input_image_size=224):
        super(ResNet_UNet_NoSkip, self).__init__()
        self.num_classes = num_classes
        self.input_image_size = input_image_size
        
        for param in ResNet.parameters():
            param.requires_grad = False
        
        self.encoder = nn.Sequential(
            ResNet.conv1,
            ResNet.bn1,
            nn.ReLU(),
            ResNet.maxpool,
            ResNet.layer1,
            ResNet.layer2,
            ResNet.layer3,
            ResNet.layer4
        )
    
        dummy_input = torch.randn(1, 3, input_image_size, input_image_size)
        x = self.encoder(dummy_input)
        
        # Define feature dimensions
        feature_dim = x.shape[1]
        half_dim = feature_dim // 2
        dim1 = feature_dim // 4
        dim2 = feature_dim // 6
        dim3 = feature_dim // 12
        dim4 = feature_dim // 24
        dim5 = feature_dim // 32
        dim6 = feature_dim // 40

        # Center
        self.center = Decoder(feature_dim, int(feature_dim // 1.5), half_dim)

        self.classification_head = nn.Sequential(
            Decoder(half_dim, dim1, dim2),
            Upsample(dim2, dim3, dim4),
            Upsample(dim4, dim5, dim6),
            Upsample(dim6, num_classes, num_classes),
            nn.Upsample(
              size=(input_image_size, input_image_size),
              mode="bilinear",
              align_corners=False,
            ),
            Conv2d(num_classes, num_classes, kernel_size=3, padding=1) # smooth output
        )

    def forward(self, image):
        x = self.encoder(image)
        x = self.center(x) 
        x = self.classification_head(x)
        return x
    
class DenseNet_UNet(Module):
    """
    - https://pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html#torchvision.models.densenet121
    - DenseNet121 has around 8 million params compared to ResNet18's 11 million, and performs a few points better on ImageNet
    - The backbone implementation is very similar to that of ResNet, as you can see below
    """

    def __init__(self, input_image_size=256):
        super(DenseNet_UNet, self).__init__()
        densenet = densenet121(weights=DenseNet121_Weights.DEFAULT) 
        for param in densenet.parameters():
            param.requires_grad = False
        
        self.layer1 = nn.Sequential(
            densenet.features.conv0,
            densenet.features.norm0,
            densenet.features.relu0,
            densenet.features.pool0,
            densenet.features.denseblock1,
            densenet.features.transition1
        )
        self.layer2 = nn.Sequential(
            densenet.features.denseblock2,
            densenet.features.transition2
        )
        self.layer3 = nn.Sequential(
            densenet.features.denseblock3, 
            densenet.features.transition3
        )
        self.layer4 = densenet.features.denseblock4 # Depth 1024 here (vs. ResNet 512)
        
        dummy_input = torch.randn(1, 3, input_image_size, input_image_size)
        x = self.layer1(dummy_input) 
        x = self.layer2(x)
        x = self.layer3(x) 
        x = self.layer4(x)
        
        # Define feature dimensions
        feature_dim = x.shape[1] 
        half_dim = feature_dim // 2
        quarter_dim = feature_dim // 4
        eighth_dim = feature_dim // 8
        sixteenth_dim = feature_dim // 16
        
        # Center
        self.center = Decoder_No_Upsample(feature_dim, int(feature_dim // 1.5), half_dim) 

        # Skip connections
        self.skip_conv1 = Conv2d(half_dim, half_dim, kernel_size=1) 
        self.skip_conv2 = Conv2d(quarter_dim, quarter_dim, kernel_size=1)
        self.skip_conv3 = Conv2d(eighth_dim, eighth_dim, kernel_size=1)

        #decoder
        self.decoder1 = Decoder(feature_dim, half_dim, quarter_dim)
        self.decoder2 = Decoder(half_dim, quarter_dim, eighth_dim)
        
        self.classification_head = nn.Sequential(
            Upsample(quarter_dim, eighth_dim, sixteenth_dim),
            Conv2d(sixteenth_dim, 1, kernel_size=2, padding=1),
            nn.Upsample(
              size=(input_image_size, input_image_size),
              mode="bilinear",
              align_corners=False,
            )
        )

    def forward(self, image):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image[:, :3, :, :]

        # Encode
        x1 = self.layer1(image)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
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
    def __init__(self, ResNet : ResNet = tv_resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1), num_classes=1, input_image_size=128):
        super(ResNet_FC, self).__init__()
        self.num_classes = num_classes
        self.input_image_size = input_image_size
        
        # Remove the fully connected layer by replacing it with an identity function
        # Use the ResNet's 'replace_fc' method if available, otherwise set to nn.Linear with matching input/output
        if hasattr(ResNet, 'fc'):
            in_features = ResNet.fc.in_features
            ResNet.fc = nn.Linear(in_features, in_features)
        self.resnet = ResNet

        dummy_input = torch.randn(1, 3, input_image_size, input_image_size)
        features = ResNet(dummy_input)
        feature_dim = features.shape[1]
        # Add a fully connected layer for classification
        self.classification_head = nn.Linear(feature_dim, input_image_size*input_image_size * num_classes)

    def forward(self, image):
        image = image[:, :3, :, :]
        x = self.resnet(image)
        x = x.flatten(start_dim=1)
        x = self.classification_head(x)
        x = x.view(-1, self.num_classes, self.input_image_size, self.input_image_size)
        return x

r"""
  _    _          _                             
 | |  | |        | |                            
 | |__| |   ___  | |  _ __     ___   _ __   ___ 
 |  __  |  / _ \ | | | '_ \   / _ \ | '__| / __|
 | |  | | |  __/ | | | |_) | |  __/ | |    \__ \
 |_|  |_|  \___| |_| | .__/   \___| |_|    |___/
                     | |                        
                     |_|                        
"""

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

    
    def forward(self, image):
        """
        Expects a numpy array of dimensions CxHxW or BxCxHxW.
        
        Accepts either a numpy array or a torch tensor.
        """
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)

        if image.max() > 1.5:
            image = image / 255.0
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        image = (image - self.mean_tensor) / self.std_tensor
        
        out = self.model.forward(image)

        out = torch.sigmoid(out)

        return (out > self.threshold).to(torch.uint8)

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
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
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
    
class Decoder_No_Upsample(nn.Module):
    """
    Helper class for the UNet architecture.
    Uses convolutional layers to upsample the input.
    Includes dropout layer to prevent overfitting.

    Removed the conv transpose layer to maintain image size.
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super(Decoder_No_Upsample, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x) -> torch.Tensor:
        return self.decoder(x)