import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Conv2d, Module
import torchvision
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import resnet18 as tv_resnet18
from transformers import SegformerForSemanticSegmentation
from torchvision.models.resnet import ResNet
from typing import Optional, Sequence

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
    def __init__(self, input_image_size=224, num_classes=1, ResNet: Optional[ResNet] = None):
        super(ResNet_UNet, self).__init__()
        self.input_image_size = input_image_size
        self.num_classes = num_classes
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

    def freeze_backbone(self, true_or_false: bool = True):
        for param in self.layer1.parameters():
            param.requires_grad = not true_or_false
        for param in self.layer2.parameters():
            param.requires_grad = not true_or_false
        for param in self.layer3.parameters():
            param.requires_grad = not true_or_false
        for param in self.layer4.parameters():
            param.requires_grad = not true_or_false

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
            densenet.features.conv0, # type: ignore
            densenet.features.norm0, # type: ignore
            densenet.features.relu0, # type: ignore
            densenet.features.pool0, # type: ignore
            densenet.features.denseblock1, # type: ignore
            densenet.features.transition1 # type: ignore
        )
        self.layer2 = nn.Sequential(
            densenet.features.denseblock2, # type: ignore
            densenet.features.transition2 # type: ignore
        )
        self.layer3 = nn.Sequential(
            densenet.features.denseblock3, # type: ignore
            densenet.features.transition3 # type: ignore
        )
        self.layer4 = densenet.features.denseblock4 # Depth 1024 here (vs. ResNet 512)
        
        dummy_input = torch.randn(1, 3, input_image_size, input_image_size)
        x = self.layer1(dummy_input) 
        x = self.layer2(x)
        x = self.layer3(x) 
        x = self.layer4(x) # type: ignore
        
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
        x4 = self.layer4(x3) # type: ignore
        
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

class SegFormer(Module):
    """
    SegFormer model for semantic segmentation from https://github.com/NVlabs/SegFormer.

    Available pretrained weights:
    - "nvidia/segformer-b0-finetuned-ade-512-512"
    - "nvidia/segformer-b1-finetuned-ade-512-512"
    - "nvidia/segformer-b2-finetuned-ade-512-512"
    - "nvidia/segformer-b3-finetuned-ade-512-512"
    - "nvidia/segformer-b4-finetuned-ade-512-512"
    - "nvidia/segformer-b5-finetuned-ade-640-640"

    - "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
    - "nvidia/segformer-b0-finetuned-cityscapes-640-1280"
    - "nvidia/segformer-b0-finetuned-cityscapes-768-768"
    - "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    - "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
    - "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
    - "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
    - "nvidia/segformer-b4-finetuned-cityscapes-1024-1024"
    - "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    
    """
    def __init__(self, num_classes=1, input_image_size=128, weights="nvidia/segformer-b2-finetuned-ade-512-512"):
        super(SegFormer, self).__init__()
        self.num_classes = num_classes
        self.input_image_size = input_image_size
        self.weights = weights

        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            weights,
            ignore_mismatched_sizes=True
        )

        for param in self.segformer.parameters():
            param.requires_grad = False

        output_feature_size = self.segformer.config.decoder_hidden_size

        # Replace the decode head to upsample to input image size
        self.segformer.decode_head.classifier = nn.Sequential( # type: ignore
        nn.ConvTranspose2d(output_feature_size, output_feature_size // 2, kernel_size=4, stride=2, padding=1), 
        nn.BatchNorm2d(output_feature_size // 2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(output_feature_size // 2, output_feature_size // 4, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(output_feature_size // 4),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_feature_size // 4, num_classes, kernel_size=3, padding=1),
    )
        
        for param in self.segformer.decode_head.classifier.parameters(): # type: ignore
            param.requires_grad = True

    def forward(self, image):
        if image.shape[1] > 3:
            image = image[:, :3, :, :]
        
        output = self.segformer(image).logits
        
        if output.shape[2] != image.shape[2] or output.shape[3] != image.shape[3]:
            output = nn.functional.interpolate(output, size=(image.shape[2], image.shape[3]), mode="bilinear", align_corners=False)
        return output
    
    def train_backbone(self, train: bool = True):
        if train:
            for param in self.segformer.parameters():
                param.requires_grad = True
        else:
            for param in self.segformer.parameters():
                param.requires_grad = False
            for param in self.segformer.decode_head.classifier.parameters(): # type: ignore
                param.requires_grad = True

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

class DeepLab(Module):
    def __init__(self, num_classes=1, input_image_size=512, backbone: str = 'resnet50'):
        """
        DeeplabV3 model for semantic segmentation using torchvision's implementation
        https://arxiv.org/abs/1706.05587

        Backbone options:
        - 'resnet50'
        - 'resnet101'
        - 'mobilenet_v3_large'
        - 'xception'
        """
        super(DeepLab, self).__init__()
        self.num_classes = num_classes
        self.input_image_size = input_image_size
        self.backbone = backbone

        if backbone == 'resnet50':
            self.deeplab = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
        elif backbone == 'resnet101':
            self.deeplab = torchvision.models.segmentation.deeplabv3_resnet101(weights='DEFAULT')
        elif backbone == 'mobilenet_v3_large':
            self.deeplab = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights='DEFAULT')
        elif backbone == 'xception':
            self.deeplab = Xception(num_classes=num_classes)
        else: # default to resnet50
            print(f"Backbone {backbone} not recognized, defaulting to resnet50")
            self.deeplab = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')

        test_input = torch.randn(1, 3, input_image_size, input_image_size)
        test_output = self.deeplab.backbone(test_input) if backbone == 'xception' else self.deeplab.backbone(test_input)['out']

        self.deeplab.classifier = DeepLabHead(test_output.shape[1], num_classes)

    def forward(self, image):
        if image.shape[1] > 3:
            image = image[:, :3, :, :]
        
        output = self.deeplab(image)
        
        if self.backbone != 'xception':
            output = output['out']

        if output.shape[2] != image.shape[2] or output.shape[3] != image.shape[3]:
            output = nn.functional.interpolate(output, size=(image.shape[2], image.shape[3]), mode="bilinear", align_corners=False)
        return output

class Xception(nn.Module):
    """
    Xception model from the paper "Xception: Deep Learning with Depthwise Separable Convolutions"
    https://arxiv.org/pdf/1610.02357v3
    
    """
    def __init__(
            self,
            num_classes: int = 1000,
    ) -> None:
        super(Xception, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            XceptionBlock(64, 128, 2, False, True, 2),
            XceptionBlock(128, 256, 2, True, True, 2),
            XceptionBlock(256, 728, 2, True, True, 2),

            XceptionBlock(728, 728, 1, True, True, 3),
            XceptionBlock(728, 728, 1, True, True, 3),
            XceptionBlock(728, 728, 1, True, True, 3),
            XceptionBlock(728, 728, 1, True, True, 3),

            XceptionBlock(728, 728, 1, True, True, 3),
            XceptionBlock(728, 728, 1, True, True, 3),
            XceptionBlock(728, 728, 1, True, True, 3),
            XceptionBlock(728, 728, 1, True, True, 3),

            XceptionBlock(728, 1024, 2, True, False, 2),

            SeparableConv2d(1024, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),

            SeparableConv2d(1536, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(2048),
            nn.ReLU(True))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Linear(2048, num_classes)
        )

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x):
        out = self.backbone(x)
        out = self.classifier(out)
        return out
    
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                stddev = float(module.stddev) if hasattr(module, "stddev") else 0.1 # type: ignore
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=stddev, a=-2, b=2)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

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
    
## DeepLab
class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int, atrous_rates: Sequence[int] = (12, 24, 36)) -> None:
        super().__init__(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )

class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: Sequence[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

#Xception blocks
class SeparableConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            **kwargs
    ) -> None:
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, groups=in_channels, bias=False, **kwargs)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                   bias=False)
    def forward(self, x):
        out = self.conv1(x)
        out = self.pointwise(out)

        return out
    
class XceptionBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            relu_first: bool,
            grow_first: bool,
            repeat_times: int,
    ) -> None:
        super(XceptionBlock, self).__init__()
        rep = []

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, stride),
                                  padding=(0, 0), bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        mid_channels = in_channels
        if grow_first:
            rep.append(nn.ReLU(True))
            rep.append(SeparableConv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            rep.append(nn.BatchNorm2d(out_channels))
            mid_channels = out_channels

        for _ in range(repeat_times - 1):
            rep.append(nn.ReLU(True))
            rep.append(SeparableConv2d(mid_channels, mid_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            rep.append(nn.BatchNorm2d(mid_channels))

        if not grow_first:
            rep.append(nn.ReLU(True))
            rep.append(SeparableConv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            rep.append(nn.BatchNorm2d(out_channels))

        if not relu_first:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(False)

        if stride != 1:
            rep.append(nn.MaxPool2d((3, 3), (stride, stride), (1, 1)))

        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        if self.skip is not None:
            identity = self.skip(x)
            identity = self.skipbn(identity)
        else:
            identity = x

        out = self.rep(x)
        out = torch.add(out, identity)

        return out
