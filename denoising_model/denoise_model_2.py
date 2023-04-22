import gpytorch
import torch.nn as nn
import numpy as np
import torch
import random


class denoise_model_2(nn.Module):
    def __init__(self, model, gp, d, device, seed, n_noise=False, residual=False):
        super(denoise_model_2, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.denoising_model = model

        self.mean_module = gpytorch.means.ConstantMean()
        covar_module_a = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5)) + \
                         gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5)) + \
                         gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        self.covar_module = covar_module_a

        self.gp = gp
        self.residual = residual
        self.norm = nn.LayerNorm(d)

        if self.gp:
            self.gp_proj_mean = nn.Linear(1, d)
            self.gp_proj_var = nn.Linear(1, d)

        self.d = d
        self.device = device
        self.n_noise = n_noise
        self.residual = residual

    def add_gp_noise(self, x, eps):

        b, s, _ = x.shape
        mean = self.mean_module(x)
        co_var = self.covar_module(x)

        dist = gpytorch.distributions.MultivariateNormal(mean, co_var)
        mean = dist.mean.unsqueeze(-1)
        co_var = dist.variance.unsqueeze(-1)

        eps_gp = self.gp_proj_mean(mean) + self.gp_proj_var(co_var) * eps
        x_noisy = x.add_(eps_gp)

        return x_noisy

    def forward(self, dec_inputs, residual=None):

        eps_dec = torch.randn_like(dec_inputs)

        if self.gp:

            dec_noisy = self.add_gp_noise(dec_inputs, eps_dec)

        elif self.n_noise:

            dec_noisy = dec_inputs

        elif self.residual:

            dec_noisy = residual

        else:
            dec_noisy = dec_inputs.add_(eps_dec * 0.05)

        dec_rec = self.denoising_model(dec_noisy)

        dec_output = self.norm(dec_rec + dec_inputs)

        return dec_output




