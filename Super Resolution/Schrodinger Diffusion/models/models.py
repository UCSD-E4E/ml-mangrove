import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn import Module
from i2sb.diffusion import Diffusion
from torch.nn import Conv2d, Module
from torchgeo.models import resnet18, get_weight
from typing import Optional

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


class Image2ImageDiffusion(Module):
    def __init__(self, num_timesteps=100, image_size=224, opt=None, resnet=resnet18(weights=get_weight("ResNet18_Weights.SENTINEL2_ALL_MOCO"))):
        super(Image2ImageDiffusion, self).__init__()
        self.opt = opt
        self.num_timesteps = num_timesteps
        self.image_size = image_size

        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            nn.ReLU(),
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

        # Infer latent space dimensions
        dummy_input = torch.randn(1, 3, self.image_size, self.image_size)
        with torch.no_grad():
            dummy_output = self.encoder(dummy_input)

        # Define feature dimensions
        feature_dim = dummy_output.shape[1]
        half_dim = feature_dim // 2
        dim1 = feature_dim // 4
        dim2 = feature_dim // 6
        dim3 = feature_dim // 12
        dim4 = feature_dim // 24
        dim5 = feature_dim // 32

        self.decoder = nn.Sequential(
            Decoder(feature_dim, half_dim, dim1),
            Upsample(dim1, dim2, dim3),
            Upsample(dim3, dim4, dim5),
            Upsample(dim5, 1, 1),
            nn.Upsample(
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        )

        self.gate_embedding = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.Sigmoid()
        )

        # Diffusion Pipeline
        self.diffuser = I2I_DiffusionLayer(4, 64)
    
    def forward(self, multispectral_image, step: Optional[int] = None):
        """
        Forward pass for the Image2ImageDiffusion model.
        
        Args:
            image (torch.Tensor): Multispectral satellite input tensor.
            step (int, optional): Current diffusion step. If None, full diffusion process will be applied.
        Returns:
            torch.Tensor: Output tensor.
        """

        # If single image, unsqueeze to add batch dimension
        if len(multispectral_image.shape) == 3:
            multispectral_image = multispectral_image.unsqueeze(0)

        # Split the input into RGB and multispectral channels
        rgb_image = multispectral_image[:, :3, :, :]  # First 3 channels for RGB

        # Encode Multispectral image
        multispectral_features = self.encoder(multispectral_image)
        del multispectral_image

        # Diffuse
        if step is None:
            # TODO: Replace with actual diffusion process
            for time_step in range(1, self.num_timesteps + 1):
                # Gate the multispectral image feature map
                gate = self.gate_embedding(torch.tensor([time_step], dtype=torch.float32).to(rgb_image.device))
                multispectral_features = self.decoder(multispectral_features * gate.unsqueeze(2).unsqueeze(3))
                rgb_image = self.diffuser(rgb_image, multispectral_features, time_step)
        else:
            gate = self.gate_embedding(torch.tensor([step], dtype=torch.float32).to(rgb_image.device))
            multispectral_features = self.decoder(multispectral_features * gate.unsqueeze(2).unsqueeze(3))
            rgb_image = self.diffuser(rgb_image, multispectral_features, step)
    
        return rgb_image


class ResNet_UNet_Diffusion(Module):
    def __init__(self, num_timesteps=1000, image_size=224, opt=None, unet=None, num_input_channels=13):
        super(ResNet_UNet_Diffusion, self).__init__()
        if unet is None:
            unet = ResNet_UNet_NoSkip(ResNet = resnet18(weights=get_weight("ResNet18_Weights.SENTINEL2_ALL_MOCO")), input_image_size=image_size, num_input_channels=num_input_channels)
        self.opt = opt
        self.input_image_size = unet.input_image_size
        self.num_timesteps = num_timesteps
        # self.image_size = image_size

        # Inherit encoder and decoder layers from ResNet UNet NoSkip
        self.encoder = unet.encoder
        self.center = unet.center
        self.decoder = unet.classification_head

        # Infer latent space dimensions
        dummy_input = torch.randn(1, num_input_channels, self.input_image_size, self.input_image_size)
        with torch.no_grad():
            dummy_output = self.encoder(dummy_input)
        output_channels = dummy_output.shape[1]
        print(f"dummy input: latent space channels={output_channels}")

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
            
            x = self.center(self.encoder(image))

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
            return self.decoder(x)
    
class ResNet_UNet_NoSkip(Module):
    """
    ResNet UNet without any skip connections.
    """
    def __init__(self, ResNet = resnet18(
                weights=get_weight("ResNet18_Weights.SENTINEL2_RGB_MOCO")
            ), num_classes=1, input_image_size=224, num_input_channels=3):
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
    
        dummy_input = torch.randn(1, num_input_channels, input_image_size, input_image_size).to(next(self.encoder.parameters()).device)
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
    
class I2I_DiffusionLayer(nn.Module):
    def __init__(self, image_size, time_embedding_dim=64):
        super(I2I_DiffusionLayer, self).__init__()
        
        # Timestep embedding network
        self.timestep_embed = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.ReLU(),
            nn.Linear(time_embedding_dim, image_size * image_size)
        )

        # Diffusion step network
        self.diffusion_step = nn.Sequential(
            nn.Conv2d(5, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 3, kernel_size=3, padding=1)
        )

    def forward(self, image, features, t):
        """
        Args:
            image (torch.Tensor): Input image tensor. [batch_size, 3, H, W]
            features (torch.Tensor): multispectral features. [batch_size, 1, H, W]
            t (torch.Tensor): Current timestep tensor (of shape [batch_size, 1]).
         """
        # Embed the timestep
        t_emb = self.timestep_embed(t.view(-1, 1))
        t_emb = t_emb.view(-1, 1, self.image_size, self.image_size)

        # Concatenate image with multispectral features and timestep embedding
        image = torch.concat((image, features), dim=1)
        image = torch.concat((image, t_emb), dim=1)
        
        # Perform the diffusion step
        return self.diffusion_step(image)

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
    def __init__(self, image_size=224, center_path='/Users/evanwu/Downloads/224_moco_resnet18_noskip.pth'):
        super(LatentSpaceExtractor, self).__init__()
        self.image_size = image_size

        PRETRAINED_CENTER_DECODER_0_WEIGHT_FIRST_VALUE = 0.6512
        RGB_MOCO_PATH = 'ResNet18_Weights.SENTINEL2_RGB_MOCO'
        ALL_MOCO_PATH = 'ResNet18_Weights.SENTINEL2_ALL_MOCO'
        PRETRAINED_WEIGHTS = torch.load(center_path, map_location=torch.device('cpu'))
        CENTER_LAYERS = ['center.decoder.0.weight', 'center.decoder.0.bias', 'center.decoder.1.weight', 'center.decoder.1.bias', 'center.decoder.1.running_mean', 'center.decoder.1.running_var', 'center.decoder.1.num_batches_tracked', 'center.decoder.4.weight', 'center.decoder.4.bias', 'center.decoder.5.weight', 'center.decoder.5.bias', 'center.decoder.5.running_mean', 'center.decoder.5.running_var', 'center.decoder.5.num_batches_tracked', 'center.decoder.7.weight', 'center.decoder.7.bias']

        missing_center_layers = [layer for layer in CENTER_LAYERS if layer not in PRETRAINED_WEIGHTS]
        if missing_center_layers:
            print('❌ Missing center layers in pretrained weights:', missing_center_layers)
        else:
            print('✅ All center layers found')

        # Build Drone RNUNNoSkip with RGB_MOCO encoder weights and loaded in center weights
        drone_RNUNNoSkip = ResNet_UNet_NoSkip(ResNet=resnet18(weights=get_weight(RGB_MOCO_PATH)), num_classes=1, input_image_size=224, num_input_channels=3)
        drone_RNUNNoSkip_base_dict = drone_RNUNNoSkip.state_dict()
        for layer in CENTER_LAYERS:
            drone_RNUNNoSkip_base_dict[layer] = PRETRAINED_WEIGHTS[layer]
        drone_RNUNNoSkip.load_state_dict(drone_RNUNNoSkip_base_dict)
        print('✅ Pretrained center layers successfully loaded into drone_RNUNNoSkip')

        # Build Satellite RNUNNoSkip with RGB_ALL encoder weights and loaded in center weights
        satellite_RNUNNoSkip = ResNet_UNet_NoSkip(ResNet=resnet18(weights=get_weight(ALL_MOCO_PATH)), num_classes=1, input_image_size=224, num_input_channels=13)
        satellite_RNUNNoSkip_base_dict = satellite_RNUNNoSkip.state_dict()
        for layer in CENTER_LAYERS:
            satellite_RNUNNoSkip_base_dict[layer] = PRETRAINED_WEIGHTS[layer]
        satellite_RNUNNoSkip.load_state_dict(satellite_RNUNNoSkip_base_dict)
        print('✅ Pretrained center layers successfully loaded into satellite_RNUNNoSkip')

        # Sanity check that the first weight value matches for satellite and drone center
        drone_state_dict = drone_RNUNNoSkip.state_dict()
        assert math.isclose(drone_state_dict['center.decoder.0.weight'].flatten()[0].item(), PRETRAINED_CENTER_DECODER_0_WEIGHT_FIRST_VALUE, abs_tol=1e-4), f"first weight from center.decoder.0.weight={drone_state_dict['center.decoder.0.weight'].flatten()[0].item()} does not equal PRETRAINED_CENTER_DECODER_0_WEIGHT_FIRST_VALUE={PRETRAINED_CENTER_DECODER_0_WEIGHT_FIRST_VALUE}"
        # print(drone_state_dict['center.decoder.0.weight'].flatten()[0].item() , 'and', PRETRAINED_CENTER_DECODER_0_WEIGHT_FIRST_VALUE)
        sat_state_dict = satellite_RNUNNoSkip.state_dict()
        assert math.isclose(sat_state_dict['center.decoder.0.weight'].flatten()[0].item(), PRETRAINED_CENTER_DECODER_0_WEIGHT_FIRST_VALUE, abs_tol=1e-4), f"first weight from center.decoder.0.weight={sat_state_dict['center.decoder.0.weight'].flatten()[0].item()} does not equal PRETRAINED_CENTER_DECODER_0_WEIGHT_FIRST_VALUE={PRETRAINED_CENTER_DECODER_0_WEIGHT_FIRST_VALUE}"
        # print(sat_state_dict['center.decoder.0.weight'].flatten()[0].item() , 'and', PRETRAINED_CENTER_DECODER_0_WEIGHT_FIRST_VALUE)

        # Assign LSE layers
        self.drone_encoder = drone_RNUNNoSkip.encoder
        self.drone_center = drone_RNUNNoSkip.center
        del drone_RNUNNoSkip

        self.satellite_encoder = satellite_RNUNNoSkip.encoder
        self.satellite_center = satellite_RNUNNoSkip.center
        del satellite_RNUNNoSkip

        # Additional sanity checks on self.drone_center and self.satellite_center
        assert math.isclose(self.drone_center.decoder[0].weight.flatten()[0].item(), PRETRAINED_CENTER_DECODER_0_WEIGHT_FIRST_VALUE, abs_tol=1e-4), f"first weight from self.drone_center.decoder={self.drone_center.decoder[0].weight.flatten()[0].item()} does not equal PRETRAINED_CENTER_DECODER_0_WEIGHT_FIRST_VALUE={PRETRAINED_CENTER_DECODER_0_WEIGHT_FIRST_VALUE}"
        print(self.drone_center.decoder[0].weight.flatten()[0].item() , 'and', PRETRAINED_CENTER_DECODER_0_WEIGHT_FIRST_VALUE)
        assert math.isclose(self.satellite_center.decoder[0].weight.flatten()[0].item(), PRETRAINED_CENTER_DECODER_0_WEIGHT_FIRST_VALUE, abs_tol=1e-4), f"first weight from self.satellite_center.decoder={self.satellite_center.decoder[0].weight.flatten()[0].item()} does not equal PRETRAINED_CENTER_DECODER_0_WEIGHT_FIRST_VALUE={PRETRAINED_CENTER_DECODER_0_WEIGHT_FIRST_VALUE}"
        print(self.satellite_center.decoder[0].weight.flatten()[0].item() , 'and', PRETRAINED_CENTER_DECODER_0_WEIGHT_FIRST_VALUE)

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
            x1 = self.drone_encoder(image)
            # x2 = self.drone_center(x1)
            return x1
        elif image.shape[1] == 13:
            # Satellite image
            x1 = self.satellite_encoder(image)
            # x2 = self.satellite_center(x1)
            return x1
        else:
            raise ValueError("Input image must have 3 (drone) or 13 (satellite) channels, got {} channels.".format(image.shape[1]))
