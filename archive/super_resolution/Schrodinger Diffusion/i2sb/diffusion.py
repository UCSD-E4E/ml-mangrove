# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

# https://github.com/NVlabs/I2SB/blob/master/i2sb/diffusion.py

import numpy as np
from tqdm import tqdm
from functools import partial
import torch

class Diffusion:
    """
    A class to perform diffusion processes for denoising diffusion probabilistic models (DDPM).

    Attributes:
        device: The device to run the computations on.
        betas: The noise schedule.
        std_fwd: Standard deviations for the forward process.
        std_bwd: Standard deviations for the backward process.
        std_sb: Standard deviations for the combined process.
        mu_x0: Coefficients for the initial state.
        mu_x1: Coefficients for the final state.
    """

    def __init__(self, betas, device):
        """
        Initializes the Diffusion class with given betas and device.

        Args:
            betas: The noise schedule.
            device: The device to run the computations on.
        """
        self.device = device

        # Compute analytic std: eq 11
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # Tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to(device)
        self.std_fwd = to_torch(std_fwd).to(device)
        self.std_bwd = to_torch(std_bwd).to(device)
        self.std_sb  = to_torch(std_sb).to(device)
        self.mu_x0 = to_torch(mu_x0).to(device)
        self.mu_x1 = to_torch(mu_x1).to(device)

    def get_std_fwd(self, step, xdim=None):
        """
        Returns the forward standard deviation for a given step.

        Args:
            step (int): The timestep.
            xdim (tuple, optional): The dimensions of the data.

        Returns:
            torch.tensor: The forward standard deviation for the given step.
        """
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def q_sample(self, step, x0, x1, ot_ode=False):
        """
        Samples an intermediate tensor x_t between the initial tensor x0 and the noisy tensor x1.

        Args:
            step (int): The timestep.
            x0 (torch.tensor): The initial tensor.
            x1 (torch.tensor): The noisy tensor.
            ot_ode (bool, optional): Whether to use ordinary differential equations for sampling. Defaults to False.

        Returns:
            torch.tensor: The sampled intermediate tensor x_t.
        """
        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0  = unsqueeze_xdim(self.mu_x0[step],  xdim)
        mu_x1  = unsqueeze_xdim(self.mu_x1[step],  xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach()

    def p_posterior(self, nprev, n, x_n, x0, ot_ode=False):
        """
        Samples p(x_{nprev} | x_n, x_0), i.e., performs a reverse diffusion step.

        Args:
            nprev (int): The previous timestep.
            n (int): The current timestep.
            x_n (torch.tensor): The tensor at the current timestep.
            x0 (torch.tensor): The initial tensor.
            ot_ode (bool, optional): Whether to use ordinary differential equations for sampling. Defaults to False.

        Returns:
            torch.tensor: The tensor at the previous timestep.
        """
        assert nprev < n
        std_n     = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

        return xt_prev

    def ddpm_sampling(self, steps, pred_x0_fn, x1, mask=None, ot_ode=False, log_steps=None, verbose=True):
        """
        Performs DDPM sampling to generate a sequence of tensors.

        Args:
            steps (list): The timesteps for sampling.
            pred_x0_fn (function): The function to predict x0.
            x1 (torch.tensor): The initial noisy tensor.
            mask (torch.tensor, optional): Mask for conditional generation. Defaults to None.
            ot_ode (bool, optional): Whether to use ordinary differential equations for sampling. Defaults to False.
            log_steps (list, optional): Steps at which to log intermediate results. Defaults to None.
            verbose (bool, optional): Whether to display progress. Defaults to True.

        Returns:
            tuple: Two tensors representing the backward trajectory and the predicted x0s.
        """
        xt = x1.detach().to(self.device)

        xs = []
        pred_x0s = []

        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            pred_x0 = pred_x0_fn(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)

            if mask is not None:
                xt_true = x1
                if not ot_ode:
                    _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                    std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                xt = (1. - mask) * xt_true + mask * xt

            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())

        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)

def unsqueeze_xdim(z, xdim):
    """
    Unsqueezes the dimensions of a tensor to match the given dimensions.

    Args:
        z (torch.tensor): The input tensor.
        xdim (tuple): The target dimensions.

    Returns:
        torch.tensor: The reshaped tensor.
    """
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]

def compute_gaussian_product_coef(sigma1, sigma2):
    """
    Computes the coefficients for the product of two Gaussian distributions.

    Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2),
    returns the parameters for the product distribution.

    Args:
        sigma1 : The standard deviation of the first Gaussian.
        sigma2 : The standard deviation of the second Gaussian.

    Returns:
        tuple: Coefficients for the product Gaussian distribution.
    """
    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var


