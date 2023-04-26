import gpytorch
import torch.nn as nn
import numpy as np
import torch
import random
from gpytorch.models import ApproximateGP
from gpytorch.variational import MeanFieldVariationalDistribution
from gpytorch.variational import VariationalStrategy


class SoftplusRBFKernel(gpytorch.kernels.RBFKernel):
    def forward(self, x1, x2, **params):
        covar = super().forward(x1, x2, **params)
        return torch.nn.functional.softplus(covar)


class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = MeanFieldVariationalDistribution(inducing_points.size(1))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution,
                                                   learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = SoftplusRBFKernel()

    def forward(self, x):

        batch_size = x.size(0)
        num_points = x.size(1)

        # Flatten the batch and num_points dimensions into a single batch dimension

        # Compute the mean and covariance using the GP model
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        # Reshape the mean and covariance to have shape (batch_size, num_points)
        mean_x = mean_x.reshape(batch_size, num_points)
        covar_x = covar_x.reshape(batch_size, num_points, num_points)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class denoise_model_2(nn.Module):
    def __init__(self, model, gp, d, device, seed, inducing_points=None, n_noise=False, residual=False):
        super(denoise_model_2, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.denoising_model = model

        self.gp = gp
        self.gp_model = GPModel(inducing_points)
        self.residual = residual
        self.norm = nn.LayerNorm(d)

        if self.gp:
            self.proj = nn.Linear(d, 4)
            self.proj_back = nn.Linear(4, d)
            self.gp_proj_mean = nn.Linear(1, d)
            self.gp_proj_var = nn.Linear(1, d)

        self.d = d
        self.device = device
        self.n_noise = n_noise
        self.residual = residual

    def add_gp_noise(self, x, eps):

        b, s, _ = x.shape

        x = self.proj(x)
        x = x.permute(1, 0, 2)
        dist = self.gp_model(x)

        mean = dist.mean.unsqueeze(-1).permute(1, 0, 2)
        co_var = dist.variance.unsqueeze(-1).permute(1, 0, 2)

        eps_gp = nn.ReLU()(self.gp_proj_mean(mean)) + nn.ReLU()(self.gp_proj_var(co_var)) * eps
        x = self.proj_back(x.permute(1, 0, 2))
        x_noisy = x.add_(eps_gp)

        return x_noisy

    def forward(self, enc_inputs, dec_inputs, residual=None):

        eps_enc = torch.randn_like(enc_inputs)
        eps_dec = torch.randn_like(dec_inputs)

        if self.gp:
            inputs = torch.cat([enc_inputs, dec_inputs], dim=1)
            eps_inputs = torch.cat([eps_enc, eps_dec], dim=1)
            input_noisy = self.add_gp_noise(inputs, eps_inputs)
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

        loss = 0

        enc_output = enc_inputs + enc_rec
        dec_output = dec_inputs + dec_rec

        return enc_output, dec_output, loss