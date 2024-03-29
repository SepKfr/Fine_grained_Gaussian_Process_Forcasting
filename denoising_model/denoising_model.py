import random

import gpytorch
import numpy as np
import torch
import torch.nn as nn
from modules.losses import normal_kl


class denoise_model(nn.Module):
    def __init__(self, gp, d, device, seed, n_noise=False, residual=False):
        super(denoise_model, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.encoder = nn.Sequential(nn.Conv1d(in_channels=d, out_channels=d, kernel_size=3, padding=int((3 - 1) / 2)),
                                     nn.Conv1d(in_channels=d, out_channels=d, kernel_size=3, padding=int((3 - 1) / 2)),
                                     nn.BatchNorm1d(d),
                                     nn.Softmax(dim=-1)).to(device)

        self.musig = nn.Linear(d, 2 * d, device=device)

        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=d, out_channels=d, kernel_size=3, padding=int((3 - 1) / 2)),
            nn.Conv1d(in_channels=d, out_channels=d, kernel_size=3, padding=int((3 - 1) / 2)),
            nn.BatchNorm1d(d),
            nn.Softmax(dim=-1), ).to(device)

        self.norm = nn.LayerNorm(d)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.mean_module_t = gpytorch.means.ConstantMean()
        self.covar_module_t = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.gp = gp
        self.residual = residual

        if self.gp:
            self.gp_proj_mean = nn.Linear(1, d)
            self.gp_proj_var = nn.Linear(1, d)

        self.d = d
        self.device = device
        self.n_noise = n_noise
        self.residual = residual

    def forward(self, x, target=None, residual=None):

        eps = torch.randn_like(x)

        if target is not None:
            s_len = target.shape[1]
            mean_t = self.mean_module_t(target)
            co_var_t = self.covar_module_t(target)

            dist = gpytorch.distributions.MultivariateNormal(mean_t, co_var_t)
            mean_t = dist.mean.unsqueeze(-1)
            co_var_t = dist.variance.unsqueeze(-1)

        if self.gp:
            b, s, _ = x.shape
            mean = self.mean_module(x)
            co_var = self.covar_module(x)

            dist = gpytorch.distributions.MultivariateNormal(mean, co_var)
            mean = dist.mean.unsqueeze(-1)
            co_var = dist.variance.unsqueeze(-1)

            eps = self.gp_proj_mean(mean) + self.gp_proj_var(co_var) * eps * 0.1
            x_noisy = x.add_(eps)

        elif self.n_noise:

            x_noisy = x

        elif self.residual:

            x_noisy = residual

        else:
            x_noisy = x.add_(eps * 0.05)

        musig = self.musig(self.encoder(x_noisy.permute(0, 2, 1)).permute(0, 2, 1))

        mu, sigma = musig[:, :, :self.d], musig[:, :, -self.d:]

        z = mu + torch.exp(sigma * 0.5) * torch.randn_like(sigma, device=self.device)

        y = self.decoder(z.permute(0, 2, 1)).permute(0, 2, 1)

        output = self.norm(y + x)

        if target is not None:

            mean_t = torch.flatten(torch.mean(mean_t, dim=-1), start_dim=1)
            co_var_t = torch.flatten(torch.mean(co_var_t, dim=-1), start_dim=1)

            mu = torch.flatten(torch.mean(mu[:, -s_len:, :], dim=-1), start_dim=1)
            sigma = torch.flatten(torch.mean(sigma[:, -s_len:, :], dim=-1), start_dim=1)

            kl_loss = normal_kl(mean_t, co_var_t, mu, sigma).mean()

        else:
            kl_loss = 0

        return output, kl_loss



