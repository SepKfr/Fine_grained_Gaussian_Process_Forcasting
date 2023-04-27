import gpytorch
import torch.nn as nn
import numpy as np
import torch
import random

class GPModel(nn.Module):
    def __init__(self):
        super(GPModel, self).__init__()

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() + gpytorch.kernels.MaternKernel(nu=2.5)
        )

    def forward(self, x):

        # Compute the mean and covariance using the GP model
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class denoise_model_2(nn.Module):
    def __init__(self, model, gp, d, device, seed, inducing_points=None, n_noise=False, residual=False):
        super(denoise_model_2, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.denoising_model = model

        self.gp = gp
        self.gp_model = GPModel()
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

        dist = self.gp_model(x)

        mean = dist.mean.unsqueeze(-1).to(self.device)
        co_var = dist.variance.unsqueeze(-1).to(self.device)

        eps_gp = nn.ReLU()(self.gp_proj_mean(mean)) + nn.ReLU()(self.gp_proj_var(co_var)) * eps
        x_noisy = x.add_(eps_gp)

        return x_noisy, dist

    def forward(self, enc_inputs, dec_inputs, residual=None):

        dist = None
        eps_enc = torch.randn_like(enc_inputs)
        eps_dec = torch.randn_like(dec_inputs)

        if self.gp:
            inputs = torch.cat([enc_inputs, dec_inputs], dim=1)
            eps_inputs = torch.cat([eps_enc, eps_dec], dim=1)
            input_noisy, dist = self.add_gp_noise(inputs, eps_inputs)
            enc_noisy = input_noisy[:, :enc_inputs.shape[1], :]
            dec_noisy = input_noisy[:, -enc_inputs.shape[1]:, :]

        elif self.n_noise:

            enc_noisy = enc_inputs
            dec_noisy = dec_inputs

        elif self.residual:

            enc_noisy = residual[0]
            dec_noisy = residual[1]

        else:
            enc_noisy = enc_inputs.add_(eps_enc * 0.1)
            dec_noisy = dec_inputs.add_(eps_dec * 0.1)

        enc_rec, dec_rec = self.denoising_model(enc_noisy, dec_noisy)

        enc_output = enc_inputs + enc_rec
        dec_output = dec_inputs + dec_rec

        return enc_output, dec_output, dist