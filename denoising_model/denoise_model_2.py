import gpytorch
import torch.nn as nn
import numpy as np
import torch
import random


class denoise_model_2(nn.Module):
    def __init__(self, model, gp, d, device, seed, nu=0.5, n_noise=False, residual=False):
        super(denoise_model_2, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.denoising_model = model

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.MaternKernel(nu=nu)

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

        eps = self.gp_proj_mean(mean) + self.gp_proj_var(co_var) * eps
        x_noisy = x.add_(eps)

        return x_noisy

    def forward(self, enc_inputs, dec_inputs, residual=None):

        eps_enc = torch.randn_like(enc_inputs)
        eps_dec = torch.randn_like(dec_inputs)

        if self.gp:

            enc_noisy = self.add_gp_noise(enc_inputs, eps_enc)
            dec_noisy = self.add_gp_noise(dec_inputs, eps_dec)

        elif self.n_noise:

            enc_noisy = enc_inputs
            dec_noisy = dec_inputs

        elif self.residual:

            enc_noisy = residual[0]
            dec_noisy = residual[1]

        else:
            enc_noisy = enc_inputs.add_(eps_enc * 0.05)
            dec_noisy = dec_inputs.add_(eps_dec * 0.05)

        enc_rec, dec_rec = self.denoising_model(enc_noisy, dec_noisy)

        loss = nn.MSELoss()(enc_rec, enc_inputs) + nn.MSELoss()(dec_rec, dec_inputs)

        enc_output = self.norm(enc_rec + enc_inputs)
        dec_output = self.norm(dec_rec + dec_inputs)

        return enc_output, dec_output, loss




