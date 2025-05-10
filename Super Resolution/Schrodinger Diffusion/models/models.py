import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Module
from i2sb.diffusion import Diffusion
from models.original_classifier import ResNet_UNet

"""
   _____   _                       _    __   _                     
  / ____| | |                     (_)  / _| (_)                    
 | |      | |   __ _   ___   ___   _  | |_   _    ___   _ __   ___ 
 | |      | |  / _` | / __| / __| | | |  _| | |  / _ \ | '__| / __|
 | |____  | | | (_| | \__ \ \__ \ | | | |   | | |  __/ | |    \__ \
  \_____| |_|  \__,_| |___/ |___/ |_| |_|   |_|  \___| |_|    |___/
                                                                   
"""       
class ResNet_UNet_Diffusion(Module):
    def __init__(self, unet = None, num_timesteps=1000, opt=None):
        super(ResNet_UNet_Diffusion, self).__init__()
        if unet is None:
            unet = ResNet_UNet()
        self.opt = opt
        self.input_image_size = unet.input_image_size
        self.num_timesteps = num_timesteps

        # Inherit encoder and decoder layers from ResNet UNet
        self.layer1 = unet.layer1
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

    def forward(self, image: torch.tensor, diffuse: bool = False, return_encoding_only: bool = False, step: int = None):
        """
        Forward pass for the ResNet UNet.
        
        Args:
            image (torch.Tensor): Input image tensor.
            diffuse (bool, optional): If True, then diffusion will be applied.
            return_encoding_only (bool, optional): Return only the latent encoding. Defaults to False.
        Returns:
            torch.Tensor: Output tensor.
        """
        # Encode
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image[:, :3, :, :]
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
                xt = x.clone()
                for i in range(self.num_timesteps):
                    time_step = torch.tensor([i], dtype=torch.float32)
                    xt = self.diffuser(xt, time_step)
                x = xt
                del xt
        
        # Classify
        if not return_encoding_only:
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
