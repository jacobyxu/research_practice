"""
Reference: https://www.youtube.com/watch?v=a4Yfz2FxXiY&list=PLtxskgdT6Ospz1aKFd5SOOAFHWoOg8DUV&index=8&t=1287s
"""

import torch
from typing import Union, Optional
from torch import nn


class NoiseScheduler():
    def __init__(
        self, n_ts: int, start: float = 0.0001, end: float = 0.02, device: Optional[str] = None,
    ):
        self.n_ts = n_ts
        self.start = start
        self.end = end
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        self.get_betas()
        self.get_alphas()

    def get_betas(self):
        self.betas = torch.linspace(self.start, self.end, self.n_ts)

    def get_alphas(self):
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = torch.nn.functional.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * \
            (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def get_index(
        self,
        vals: Union[torch.FloatTensor, torch.cuda.FloatTensor],
        t: Union[torch.IntTensor, torch.cuda.IntTensor],
        x_shape: torch.Size,
    ):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        index = out.reshape(
            batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
        return index

    def get_betas_t(
        self,
        t: Union[torch.IntTensor, torch.cuda.IntTensor],
        x_shape: torch.Size,
    ):
        betas_t = self.get_index(self.betas, t, x_shape)
        return betas_t

    def get_sqrt_alphas_cumprod_t(
        self,
        t: Union[torch.IntTensor, torch.cuda.IntTensor],
        x_shape: torch.Size,
    ):
        sqrt_alphas_cumprod_t = self.get_index(
            self.sqrt_alphas_cumprod, t, x_shape)
        return sqrt_alphas_cumprod_t

    def get_sqrt_one_minus_alphas_cumprod_t(
        self,
        t: Union[torch.IntTensor, torch.cuda.IntTensor],
        x_shape: torch.Size,
    ):
        sqrt_one_minus_alphas_cumprod_t = self.get_index(
            self.sqrt_one_minus_alphas_cumprod, t, x_shape
        )
        return sqrt_one_minus_alphas_cumprod_t

    def get_sqrt_recip_alphas_t(
        self,
        t: Union[torch.IntTensor, torch.cuda.IntTensor],
        x_shape: torch.Size,
    ):
        sqrt_recip_alphas_t = self.get_index(
            self.sqrt_recip_alphas, t, x_shape
        )
        return sqrt_recip_alphas_t

    def get_posterior_variance_t(
        self,
        t: Union[torch.IntTensor, torch.cuda.IntTensor],
        x_shape: torch.Size,
    ):
        posterior_variance_t = self.get_index(
            self.posterior_variance, t, x_shape
        )
        return posterior_variance_t

    def forward_diffusion_sample(
        self,
        x_0: Union[torch.FloatTensor, torch.cuda.FloatTensor],
        t: Union[torch.IntTensor, torch.cuda.IntTensor],
    ):
        """ 
        Takes an image and a timestep as input (pixle value in [-1, 1]) and 
        returns the noisy version of it
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_sqrt_alphas_cumprod_t(
            t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_sqrt_one_minus_alphas_cumprod_t(
            t, x_0.shape
        )
        # mean + variance
        x_t = (
            sqrt_alphas_cumprod_t.to(self.device) * x_0.to(self.device)
            + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device)
        )
        return x_t.to(self.device), noise.to(self.device)


def get_diffusion_loss(
    model: nn.Module,
    noisescheduler: NoiseScheduler,
    x_0: Union[torch.FloatTensor, torch.cuda.FloatTensor],
    t: torch.Tensor,
):
    x_noisy, noise = noisescheduler.forward_diffusion_sample(x_0, t)
    noise_pred = model(x_noisy, t)
    loss = torch.nn.functional.l1_loss(noise, noise_pred)
    return loss
