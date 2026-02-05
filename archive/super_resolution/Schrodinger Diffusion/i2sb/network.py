# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import pickle
import torch

from guided_diffusion.script_util import create_model

ADM_IMG256_UNCOND_CKPT = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
I2SB_IMG256_UNCOND_PKL = "256x256_diffusion_uncond_fixedsigma.pkl"
I2SB_IMG256_UNCOND_CKPT = "256x256_diffusion_uncond_fixedsigma.pt"
I2SB_IMG256_COND_PKL = "256x256_diffusion_cond_fixedsigma.pkl"
I2SB_IMG256_COND_CKPT = "256x256_diffusion_cond_fixedsigma.pt"

class Image256Net(torch.nn.Module):
    def __init__(self, noise_levels, use_fp16=False, cond=False, pretrained_adm=True, ckpt_dir="data/"):
        super(Image256Net, self).__init__()

        # initialize model
        ckpt_pkl = os.path.join(ckpt_dir, I2SB_IMG256_COND_PKL if cond else I2SB_IMG256_UNCOND_PKL)
        with open(ckpt_pkl, "rb") as f:
            kwargs = pickle.load(f)
        kwargs["use_fp16"] = use_fp16
        self.diffusion_model = create_model(**kwargs) # TODO:Insert our model here
        # load (modified) adm ckpt
        if pretrained_adm:
            ckpt_pt = os.path.join(ckpt_dir, I2SB_IMG256_COND_CKPT if cond else I2SB_IMG256_UNCOND_CKPT)
            out = torch.load(ckpt_pt, map_location="cpu")
            self.diffusion_model.load_state_dict(out)

        self.diffusion_model.eval()
        self.cond = cond
        self.noise_levels = noise_levels

    def forward(self, x, steps, cond=None):

        t = self.noise_levels[steps].detach()
        assert t.dim()==1 and t.shape[0] == x.shape[0]

        x = torch.cat([x, cond], dim=1) if self.cond else x
        return self.diffusion_model(x, t)