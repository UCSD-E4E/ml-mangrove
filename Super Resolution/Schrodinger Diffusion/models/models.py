import torch
import torch.nn as nn
import numpy as np
from torch.nn import Module
from i2sb.diffusion import Diffusion
from torch.nn import Conv2d, Module
from torchgeo.models import resnet18, get_weight

"""
Pretrained model Weights from SSL4EO-12 dataset
@ https://github.com/zhu-xlab/SSL4EO-S12

Imported using torchgeo
@ https://torchgeo.readthedocs.io/en/stable/api/models.html

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
class ResNet_UNet_Diffusion(Module):
    def __init__(self, num_timesteps=1000, image_size=224, opt=None, unet=None):
        super(ResNet_UNet_Diffusion, self).__init__()
        if unet is None:
            unet = ResNet_UNet(ResNet = resnet18(weights=get_weight("ResNet18_Weights.SENTINEL2_ALL_MOCO")), image_size=image_size, num_input_channels=13)
        self.opt = opt
        self.input_image_size = unet.input_image_size
        self.num_timesteps = num_timesteps
        self.image_size = image_size

        # Inherit encoder and decoder layers from ResNet UNet
        self.layer1 = unet.layer1
        # print("conv1 weight shape:", self.layer1[0].weight.shape)
        self.layer2 = unet.layer2
        self.layer3 = unet.layer3
        self.layer4 = unet.layer4
        self.center = unet.center
        self.skip_conv1 = unet.skip_conv1
        self.skip_conv2 = unet.skip_conv2
        self.skip_conv3 = unet.skip_conv3
        self.decoder1 = unet.decoder1
        self.decoder2 = unet.decoder2
        self.classification_head = unet.classification_head

        # Infer latent space dimensions
        dummy_input = torch.randn(1, 3, self.input_image_size, self.input_image_size)
        with torch.no_grad():
            dummy_output = self.center(self.layer4(self.layer3(self.layer2(self.layer1(dummy_input)))))
        output_channels = dummy_output.shape[1]

        # Diffusion Pipeline
        self.diffuser = DiffusionLayer(output_channels, 64)

    def forward(self, image: torch.tensor, diffuse: bool = True, return_encoding_only: bool = False, step: int = None, latent_input: bool = False):
        """
        Forward pass for the ResNet UNet.
        
        Args:
            image (torch.Tensor): Input tensor.
            diffuse (bool, optional): If True, then diffusion will be applied.
            return_encoding_only (bool, optional): Return only the latent encoding. Defaults to False.
            step (int, optional): Current diffusion step. If None, all steps will be applied.
            latent_input (bool, optional): If True, the input is already in the latent space. Defaults to False.
        Returns:
            torch.Tensor: Output tensor.
        """

        if latent_input:
            x = image
        else:
            # Encode image
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            x1 = self.layer1(image)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            x = self.center(x4)

        # Diffuse
        if diffuse:
            if step is not None:
                x = self.diffuser(x, step)
            else:
                for i in range(1, self.num_timesteps + 1):
                    # instead of having time_step be a [1] length array of i, we might want it to be a [B] length array of i
                    # time_step = x.new_full((x.size(0),), fill_value=float(i))
                    time_step = torch.tensor([i], dtype=torch.float32, device=x.device)
                    x = self.diffuser(x, time_step)
        
        if return_encoding_only:
            return x
        else:
            # Classify
            x = torch.cat((x, self.skip_conv1(x3)), dim=1)
            x = self.decoder1(x)
            x = torch.cat((x, self.skip_conv2(x2)), dim=1)
            x = self.decoder2(x)
            x = torch.cat((x, self.skip_conv3(x1)), dim=1)
            x = self.classification_head(x)
            return x
    
class ResNet_UNet(Module):
    """
    UNet architecture with ResNet encoder.
    """
    def __init__(self, ResNet = None, image_size=224, num_input_channels=3):
        super(ResNet_UNet, self).__init__()
        self.input_image_size= image_size
        if ResNet is None:
            ResNet = resnet18(
                weights=get_weight("ResNet18_Weights.SENTINEL2_RGB_SECO")
            )
        
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

        dummy_input = torch.randn(1, 3, self.input_image_size, self.input_image_size)
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
            Conv2d(sixteenth_dim, 1, kernel_size=2, padding=1),
            nn.Upsample(
              size=(self.input_image_size, self.input_image_size),
              mode="bilinear",
              align_corners=False,
            ),
            Conv2d(1, 1, kernel_size=3, padding=1) # smooth output
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

class DiffusionLayer(nn.Module):
    def __init__(self, latent_dim, time_embedding_dim):
        super(DiffusionLayer, self).__init__()
        # Timestep embedding network
        self.timestep_embed = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.ReLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        # Diffusion step network
        self.diffusion_step = nn.Sequential(
            nn.Conv2d(latent_dim + time_embedding_dim, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1)
        )

    def forward(self, latent, t):
        """
        Args:
            latent (torch.Tensor): Latent space tensor.
            t (torch.Tensor): Current timestep tensor (of shape [batch_size, 1]).
         """
        # Embed the timestep
        t = t.to(torch.float32)
        t_emb = self.timestep_embed(t.view(-1, 1))
        t_emb = t_emb.unsqueeze(2).unsqueeze(3).expand(-1, -1, latent.shape[2], latent.shape[3])
        
        # Concatenate timestep embedding with the latent space
        latent = torch.cat((latent, t_emb), dim=1)
        # Perform the diffusion step
        return self.diffusion_step(latent)
    


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
    

class LatentSpaceExtractor(Module):
    """
    Helper class to extract latent space from images.

    Use this when preprocessing a dataset for training the diffusion model.
    """
    def __init__(self, drone_resnet=None, satellite_resnet=None, image_size=224):
        super(LatentSpaceExtractor, self).__init__()
        self.image_size = image_size

        # Build Drone Encoder
        drone_ResNet_UNet = ResNet_UNet(ResNet=drone_resnet, image_size=image_size, num_input_channels=3)
        self.drone_layer1 = drone_ResNet_UNet.layer1
        self.drone_layer2 = drone_ResNet_UNet.layer2
        self.drone_layer3 = drone_ResNet_UNet.layer3
        self.drone_layer4 = drone_ResNet_UNet.layer4
        self.drone_center = drone_ResNet_UNet.center
        del drone_ResNet_UNet

        # Build Satellite Encoder
        if satellite_resnet is None:
            sat_ResNet_UNet = ResNet_UNet(resnet18(weights=get_weight("ResNet18_Weights.SENTINEL2_ALL_MOCO")), image_size=image_size, num_input_channels=13)
        else:
            sat_ResNet_UNet = ResNet_UNet(ResNet=satellite_resnet, image_size=image_size, num_input_channels=13)
        
        self.sat_layer1 = sat_ResNet_UNet.layer1
        self.sat_layer2 = sat_ResNet_UNet.layer2
        self.sat_layer3 = sat_ResNet_UNet.layer3
        self.sat_layer4 = sat_ResNet_UNet.layer4
        self.sat_center = sat_ResNet_UNet.center
        del sat_ResNet_UNet


    def forward(self, image) -> torch.Tensor:
        """
        Forward pass to extract latent space.
        """
        # If single image, unsqueeze to add batch dimension
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Check if images are 3-channel RGB (drone) or 13-channel Multispectral (satellite)
        if image.shape[1] == 3:
            # Drone image
            x1 = self.drone_layer1(image)
            x2 = self.drone_layer2(x1)
            x3 = self.drone_layer3(x2)
            x4 = self.drone_layer4(x3)
            x4 = self.drone_center(x4)
            return x4
        elif image.shape[1] == 13:
            # Satellite image
            x1 = self.sat_layer1(image)
            x2 = self.sat_layer2(x1)
            x3 = self.sat_layer3(x2)
            x4 = self.sat_layer4(x3)
            x4 = self.sat_center(x4)
            return x4
        else:
            raise ValueError("Input image must have 3 (drone) or 13 (satellite) channels, got {} channels.".format(image.shape[1]))
