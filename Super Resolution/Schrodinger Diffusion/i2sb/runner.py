# Code is a heavily modified version of https://github.com/NVlabs/I2SB/blob/master/i2sb/runner.py
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch_ema import ExponentialMovingAverage
from torch.utils.data import DataLoader
import torchvision.utils as tu
from models.models import ResNet_UNet_Diffusion, ResNet_UNet
from . import util
from .diffusion import Diffusion
from typing import Tuple
from tqdm import tqdm

def build_optimizer_sched(opt, net):
    """
    Builds the optimizer and learning rate scheduler for training.

    Args:
        opt: Configuration options.
        net: Neural network model.

    Returns:
        Tuple of (optimizer, scheduler)
    """
    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    
    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Loaded 'optimizer' from checkpoint path")
        else:
            print(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            print("Loaded 'sched' from checkpoint path")
        else:
            print(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched

def make_beta_schedule(opt):
    """
    Get a pre-defined beta schedule for the given name.
    
    Schedules:
    - linear: linear schedule from Ho et al (2020).
    - i2sb: I2SB beta schedule.
    - cosine: cosine schedule from Nichol et al (2021).
    """
    schedule_name=opt.beta_schedule
    n_timestep=opt.interval
    linear_start=1e-4
    linear_end=opt.beta_max / opt.interval
    # Linear beta schedule from Ho et al. (2020)
    # https://arxiv.org/abs/2006.11239
    if schedule_name == "linear":
        scale = 1000 / n_timestep
        start = scale * linear_start
        end = scale * linear_end
        betas = np.linspace(start, end, n_timestep)
    # I2SB beta schedule
    elif schedule_name == "i2sb":
        betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        ).numpy()
    # Cosine beta schedule from Nichol et al. (2021)
    # https://arxiv.org/abs/2102.09672
    elif schedule_name == "cosine":
        betas = betas_for_alpha_bar(
            n_timestep,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise ValueError(f"Unknown beta schedule {schedule_name=}")
    
    betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
    return betas

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class Runner(object):
    def __init__(self, opt):
        """
        Initializes the Runner class.

        Args:
            opt: Configuration options.
        """
        super(Runner, self).__init__()
        self.writer = util.build_log_writer(opt)
        betas = make_beta_schedule(opt)
        self.diffusion = Diffusion(betas, opt.device)
        self.noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval

        # Build Model
        unet = ResNet_UNet()
        # change 1: if opt.unet_path: unet = unet.load_state_dict(torch.load(opt.unet_path)) 
        self.net = ResNet_UNet_Diffusion(unet=unet, num_timesteps=len(betas))
        # print("conv1 weight shape:", self.net.layer1[0].weight.shape)
        # print("betas length:", len(betas))

        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)
        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            self.ema.load_state_dict(checkpoint["ema"])
            print("Loaded 'net' and 'ema' from checkpoint path")
        self.net.to(opt.device)
        self.ema.to(opt.device)
        print(f"Built {opt.diffusion_type} Diffusion Model with {len(betas)} steps and {opt.beta_schedule} beta schedule!")
    
    # train using algorithm 1 described in paper
    def train(self, opt, train_dataset, val_dataset):
        """
        Trains the network using the specified datasets.

        Args:
            opt: Configuration options.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
        """
        train_loss = [] # store avg loss per iteration
        evaluate_loss = [] # store (iteration, avg loss) only when eval occurs

        optimizer, sched = build_optimizer_sched(opt, self.net)
        train_loader = util.setup_loader(train_dataset, opt.microbatch)
        val_loader = util.setup_loader(val_dataset, opt.microbatch)

        self.net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for iteration in range(opt.num_itr):
            total_train_loss = 0
            optimizer.zero_grad()
            print_metrics = (iteration < 10) or (iteration % 50 == 0)
            for _ in tqdm(range(n_inner_loop), desc="train_inner_loop", disable=not print_metrics):
                # sample from boundary pair
                x0, x1 = self.sample_batch(opt, train_loader)
                step = torch.randint(0, opt.interval, (x0.shape[0],)).to(opt.device) 
                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode).to(opt.device) # intermediate noisy image

                # predict diffusion step
                pred = self.net(xt, diffuse=True, return_encoding_only=True, step=step, latent_input=True) # predicted noise
                label = self.compute_label(step, x0, xt) # ground truth noise
                # if iteration == 0:
                #   print("pred shape", pred.shape)
                #   print("label shape", label.shape)
                loss = F.mse_loss(pred, label)
                loss.backward()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / n_inner_loop
            train_loss.append(avg_train_loss)

            optimizer.step()
            self.ema.update()
            if sched is not None:
                sched.step()

            # -------- logging --------
            if print_metrics:
              print("train_it {}/{} | lr:{} | noise_prediction_loss:{}".format(
                  1 + iteration,
                  opt.num_itr,
                  "{:.2e}".format(optimizer.param_groups[0]['lr']),
                  "{:+.4f}".format(avg_train_loss),
              ))
            if iteration % 1000 == 0 or iteration == opt.num_itr - 1:
                ckpt_name = f"model_{iteration:06d}.pt"
                ckpt_path = opt.ckpt_path / ckpt_name
                torch.save({
                    "net": self.net.state_dict(),
                    "ema": self.ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "sched": sched.state_dict() if sched is not None else sched,
                }, ckpt_path)

                train_loss_path = opt.ckpt_path / "train_losses.npy"
                eval_metrics_path = opt.ckpt_path / "eval_losses.npy"
                np.save(train_loss_path, np.array(train_loss, dtype=np.float32))
                np.save(eval_metrics_path, np.array(evaluate_loss, dtype=np.float32))
                print(f"Saved weights and losses on iteration {iteration}")

            if iteration % 200 == 0:  # 0, 0.5k, 3k, 6k, 9k
                self.net.eval()
                avg_noise_prediction_loss, avg_reconstruction_loss = self.evaluation(opt, iteration, val_loader)
                evaluate_loss.append((iteration, avg_noise_prediction_loss, avg_reconstruction_loss))
                self.net.train()

        train_loader.close()
        val_loader.close()

        return train_loss, evaluate_loss

    # evaluate on both algorithm 1 and algorithm 2 described in paper
    @torch.no_grad()
    def evaluation(self, opt, it, val_loader): # evaluation of self.net with loss calculation
        total_noise_prediction_loss = 0
        total_reconstruction_loss = 0
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)

        for _ in tqdm(range(n_inner_loop), desc="eval_inner_loop"):
            high_res_image, low_res_image = self.sample_batch(opt, val_loader)
            x0 = high_res_image.to(opt.device)
            x1 = low_res_image.to(opt.device)

            # evaluate model noise prediction
            step = torch.randint(0, opt.interval, (x0.shape[0],)).to(opt.device) 
            xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode).to(opt.device) # intermediate noisy image
            pred_noise = self.net(xt, diffuse=True, return_encoding_only=True, step=step, latent_input=True) # predicted noise
            label_noise = self.compute_label(step, x0, xt) # ground truth noise
            noise_prediction_loss = F.mse_loss(pred_noise, label_noise)
            total_noise_prediction_loss += noise_prediction_loss.item()

            # evaluate reconstruction with ddpm
            xs, pred_x0s = self.ddpm_sampling(opt, x1, clip_denoise=opt.clip_denoise, verbose=False)
            x0_hat = pred_x0s[:, 0].to(opt.device)
            reconstrution_loss = F.mse_loss(x0_hat, x0)
            total_reconstruction_loss += reconstrution_loss.item()
          
        avg_noise_prediction_loss = total_noise_prediction_loss / n_inner_loop
        avg_reconstruction_loss = total_reconstruction_loss / n_inner_loop
        print("EVALUATE: eval_it {}/{} | noise_prediction_loss:{} | reconstruction_loss:{}".format(
                  1 + it,
                  opt.num_itr,
                  "{:+.4f}".format(avg_noise_prediction_loss),
                  "{:+.4f}".format(avg_reconstruction_loss),
              ))

        torch.cuda.empty_cache()
        return avg_noise_prediction_loss, avg_reconstruction_loss
    
    def compute_label(self, step: int, x0: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        """
        Computes the label for training as per Equation 12.

        Args:
            step (int): The timestep.
            x0 (torch.tensor): The initial tensor.
            xt (torch.tensor): The tensor at timestep t.
        Returns:
            torch.tensor: The computed label.
        """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(self, step: int, xt: torch.Tensor, net_out: torch.Tensor, clip_denoise=False) -> torch.Tensor:
        """
        Args:
            step (int): The timestep.
            xt (torch.tensor): The tensor at timestep t.
            net_out (torch.tensor): The network output.
            clip_denoise (bool): Whether to clip the denoised output. Defaults to False.

        Returns:
            torch.tensor: The recovered x0.
        """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def sample_batch(self, opt, loader, scale_factor=0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a batch of data.

        Args:
            opt: Configuration options.
            loader: Data loader.
        Returns:
            tuple: (high_res_img, low_res_img)
        """
        if opt.diffusion_type == "schrodinger_bridge":
            low_res_img, high_res_img = next(loader)
            with torch.no_grad():
                high_res_img = high_res_img.to(opt.device)
                high_res_size = high_res_img.shape[-2:]
                # upscale the low res image to match the high res image
                low_res_img = F.interpolate(low_res_img, high_res_size, mode='bilinear', align_corners=False)
        else:  
            clean_img = next(loader)
            with torch.no_grad():
                high_res_img = clean_img.to(opt.device)
                orig_size = opt.image_size
                # Downscale the image
                low_res_img = F.interpolate(high_res_img, scale_factor=scale_factor, mode='bilinear', align_corners=False)
                # Upscale back to original size
                low_res_img = F.interpolate(low_res_img, size=orig_size, mode='bilinear', align_corners=False)
        
        x0 = high_res_img.detach().to(opt.device)
        x1 = low_res_img.detach().to(opt.device)

        assert x0.shape == x1.shape
        return x0, x1
    
    @torch.no_grad()
    def ddpm_sampling(self, opt, x1, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True):
        """
        Performs DDPM sampling to generate a sequence of tensors.

        Args:
            opt: Configuration options.
            x1 (torch.tensor): The initial noisy tensor.
            mask (torch.tensor, optional): Mask for conditional generation. Defaults to None.
            cond (torch.tensor, optional): Condition tensor. Defaults to None.
            clip_denoise (bool, optional): Whether to clip the denoised output. Defaults to False.
            nfe (int, optional): Number of function evaluations. Defaults to None.
            log_count (int, optional): Number of log steps. Defaults to 10.
            verbose (bool, optional): Whether to display progress. Defaults to True.

        Returns:
            tuple: Two tensors representing the backward trajectory and the predicted x0s.
        """
        nfe = nfe or opt.interval - 1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe + 1)

        log_count = min(len(steps) - 1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps) - 1, log_count)]
        assert log_steps[0] == 0
        # print(f"log_steps for ddpm_sampling: {log_steps}")

        x1 = x1.to(opt.device)
        if cond is not None:
            cond = cond.to(opt.device)
        if mask is not None:
            mask = mask.to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):
                step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                out = self.net(xt, diffuse=True, return_encoding_only=True, step=step, latent_input=True)
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose,
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return xs, pred_x0

