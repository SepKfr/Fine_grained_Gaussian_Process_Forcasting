import gpytorch
import torch.nn as nn
import numpy as np
import torch
import random
from denoising_model.DeepGP import DeepGPp
from modules.feedforward import PoswiseFeedForwardNet

torch.autograd.set_detect_anomaly(True)


class denoise_model_2(nn.Module):
    def __init__(self, nu, model, gp, d, device, seed, n_noise=False, residual=False):
        super(denoise_model_2, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.denoising_model = model
        if gp:
            self.deep_gp = DeepGPp(nu, d, seed)
            self.proj_up = nn.Linear(16, d)
        self.gp = gp

        self.residual = residual
        self.norm = nn.LayerNorm(d)
        self.norm1 = nn.LayerNorm(d)

        self.d = d
        self.device = device
        self.n_noise = n_noise
        self.residual = residual
        self.ffn = PoswiseFeedForwardNet(
            d_model=d, d_ff=d*4, seed=seed)

    def add_gp_noise(self, x):

        b, s, _ = x.shape

        dist = self.deep_gp(x)

        eps_gp = dist.sample().permute(1, 2, 0)

        x_noisy = self.norm1(x + self.proj_up(eps_gp))

        return x_noisy, dist

    def forward(self, enc_inputs, dec_inputs):

        eps_enc = torch.randn_like(enc_inputs)
        eps_dec = torch.randn_like(dec_inputs)
        dist_dec = None

        if self.gp:

            enc_noisy, dist_enc = self.add_gp_noise(enc_inputs)
            dec_noisy, dist_dec = self.add_gp_noise(dec_inputs)

        elif self.n_noise:

            enc_noisy = enc_inputs
            dec_noisy = dec_inputs

        else:
            enc_noisy = enc_inputs.add_(eps_enc * 0.05)
            dec_noisy = dec_inputs.add_(eps_dec * 0.05)

        enc_rec, dec_rec = self.denoising_model(enc_noisy, dec_noisy)

        dec_output = self.norm(dec_inputs + self.ffn(dec_rec))

        return dec_output, dist_dec