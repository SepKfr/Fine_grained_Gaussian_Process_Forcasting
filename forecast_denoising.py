import random

import numpy as np
import torch
import torch.nn as nn
from denoising_model.denoise_model_2 import denoise_model_2
from forecasting_models.LSTM import RNN
from modules.transformer import Transformer

import warnings
warnings.filterwarnings("ignore")


class denoising_layer(nn.Module):

    def __init__(self, d, device, seed):
        super(denoising_layer, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.d = d
        self.device = device

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

    def forward(self, x_noisy):

        musig = self.musig(self.encoder(x_noisy.permute(0, 2, 1)).permute(0, 2, 1))

        mu, sigma = musig[:, :, :self.d], musig[:, :, -self.d:]

        z = mu + torch.exp(sigma * 0.5) * torch.randn_like(sigma, device=self.device)

        y = self.decoder(z.permute(0, 2, 1)).permute(0, 2, 1)

        return y


class Denoising(nn.Module):
    def __init__(self, d, device, seed):
        super(Denoising, self).__init__()

        self.denoising_enc = denoising_layer(d, device, seed)
        self.denoising_dec = denoising_layer(d, device, seed)

    def forward(self, x_en, x_de):

        out_en = self.denoising_enc(x_en)
        out_de = self.denoising_dec(x_de)

        return out_en, out_de


class Forecast_denoising(nn.Module):
    def __init__(self, model_name:str, config: tuple, gp: bool,
                 denoise: bool, device: torch.device,
                 seed: int, pred_len: int, attn_type: str,
                 no_noise: bool, residual: bool):

        super(Forecast_denoising, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        src_input_size, tgt_input_size, d_model, n_heads, d_k, stack_size = config

        self.pred_len = pred_len
        self.denoise_model = Denoising(d_model, device, seed)

        if "LSTM" in model_name:

            self.forecasting_model = RNN(n_layers=stack_size,
                                         hidden_size=d_model,
                                         src_input_size=src_input_size,
                                         device=device,
                                         d_r=0,
                                         seed=seed,
                                         pred_len=pred_len)

        else:

            self.forecasting_model = Transformer(src_input_size=src_input_size,
                                                 tgt_input_size=tgt_input_size,
                                                 pred_len=pred_len,
                                                 d_model=d_model,
                                                 d_ff=d_model * 4,
                                                 d_k=d_k, d_v=d_k, n_heads=n_heads,
                                                 n_layers=stack_size, src_pad_index=0,
                                                 tgt_pad_index=0, device=device,
                                                 attn_type=attn_type,
                                                 seed=seed)

        self.de_model = denoise_model_2(self.denoise_model, gp, d_model, device, seed, n_noise=no_noise, residual=residual)
        self.denoise = denoise
        self.residual = residual
        self.final_projection = nn.Linear(d_model, 1)
        self.enc_embedding = nn.Linear(src_input_size, d_model)
        self.dec_embedding = nn.Linear(tgt_input_size, d_model)

    def forward(self, enc_inputs, dec_inputs):

        enc_inputs = self.enc_embedding(enc_inputs)
        dec_inputs = self.dec_embedding(dec_inputs)

        enc_outputs, dec_outputs = self.forecasting_model(enc_inputs, dec_inputs)

        loss = 0.0

        if self.residual:
            residual = [enc_outputs - enc_inputs, dec_outputs - dec_inputs]
        else:
            residual = None

        if self.denoise:
            enc_outputs, dec_outputs, loss = self.de_model(enc_outputs.clone(), dec_outputs.clone(), residual)

        outputs = self.final_projection(dec_outputs[:, -self.pred_len:, :])
        return outputs, loss
