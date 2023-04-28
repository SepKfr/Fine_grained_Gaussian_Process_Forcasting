import gpytorch
import torch.nn as nn
import numpy as np
import torch
import random
from denoising_model.DeepGP import DeepGPp
torch.autograd.set_detect_anomaly(True)


class denoise_model_2(nn.Module):
    def __init__(self, model, gp, d, device, seed, train_x_shape, n_noise=False, residual=False):
        super(denoise_model_2, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.denoising_model = model
        self.deep_gp = DeepGPp(train_x_shape, d)
        self.gp = gp
        self.mean_proj = nn.Linear(1, d)

        self.residual = residual
        self.norm = nn.LayerNorm(d)

        self.d = d
        self.device = device
        self.n_noise = n_noise
        self.residual = residual

    def add_gp_noise(self, x, eps):

        b, s, _ = x.shape

        dist = self.deep_gp(x)
        eps_gp = dist.sample()

        eps_gp = nn.ReLU()(self.mean_proj(eps_gp.permute(1, 2, 0)))

        x_noisy = x + eps_gp

        return x_noisy, dist

    def forward(self, enc_inputs, dec_inputs, residual=None):

        eps_enc = torch.randn_like(enc_inputs)
        eps_dec = torch.randn_like(dec_inputs)
        dist = None

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