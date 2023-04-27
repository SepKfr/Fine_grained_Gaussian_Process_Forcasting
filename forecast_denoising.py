import random

import numpy as np
import torch
import torch.nn as nn
from denoising_model.denoise_model_2 import denoise_model_2
from forecasting_models.LSTM import RNN
from modules.transformer import Transformer

import warnings
warnings.filterwarnings("ignore")


class Forecast_denoising(nn.Module):
    def __init__(self, model_name:str, config: tuple, gp: bool,
                 denoise: bool, device: torch.device,
                 seed: int, pred_len: int, attn_type: str,
                 no_noise: bool, residual: bool, mean_var_gp):

        super(Forecast_denoising, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        src_input_size, tgt_input_size, d_model, n_heads, d_k, stack_size = config

        self.pred_len = pred_len

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
                                                 seed=seed,
                                                 )

        self.de_model = denoise_model_2(self.forecasting_model, gp,
                                        d_model, device, seed,
                                        n_noise=no_noise,
                                        residual=residual,
                                        mean_var_gp=mean_var_gp)
        self.denoise = denoise
        self.residual = residual
        self.final_projection = nn.Linear(d_model, 1)
        self.enc_embedding = nn.Linear(src_input_size, d_model)
        self.dec_embedding = nn.Linear(tgt_input_size, d_model)

    def forward(self, enc_inputs, dec_inputs):

        enc_inputs = self.enc_embedding(enc_inputs)
        dec_inputs = self.dec_embedding(dec_inputs)

        enc_outputs, dec_outputs = self.forecasting_model(enc_inputs, dec_inputs)

        if self.residual:
            residual = [enc_outputs - enc_inputs, dec_outputs - dec_inputs]
        else:
            residual = None

        if self.denoise:
            enc_outputs, dec_outputs = self.de_model(enc_outputs.clone(), dec_outputs.clone(), residual)

        outputs = self.final_projection(dec_outputs[:, -self.pred_len:, :])
        return outputs
