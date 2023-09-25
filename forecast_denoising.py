import random
import gpytorch
import numpy as np
import torch
import torch.nn as nn
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from denoising_model.denoise_model_2 import denoise_model_2
from forecasting_models.LSTM import RNN
from modules.feedforward import PoswiseFeedForwardNet
from modules.transformer import Transformer
torch.autograd.set_detect_anomaly(True)


class Forecast_denoising(nn.Module):
    def __init__(self, nu: float, model_name: str, config: tuple, gp: bool,
                 denoise: bool, device: torch.device,
                 seed: int, pred_len: int, attn_type: str,
                 no_noise: bool, residual: bool, input_corrupt: bool):

        super(Forecast_denoising, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        src_input_size, tgt_input_size, d_model, n_heads, d_k, stack_size = config

        self.pred_len = pred_len
        self.input_corrupt = input_corrupt

        if "LSTM" in model_name:

            self.forecasting_model = RNN(n_layers=stack_size,
                                         hidden_size=d_model,
                                         src_input_size=src_input_size,
                                         device=device,
                                         d_r=0,
                                         seed=seed,
                                         pred_len=pred_len)

        else:

            self.forecasting_model = Transformer(pred_len=pred_len,
                                                 d_model=d_model,
                                                 d_ff=d_model * 4,
                                                 d_k=d_k, d_v=d_k, n_heads=n_heads,
                                                 n_layers=stack_size, device=device,
                                                 attn_type=attn_type,
                                                 seed=seed,
                                                 )

        self.de_model = denoise_model_2(nu,
                                        self.forecasting_model, gp,
                                        d_model, device, seed,
                                        n_noise=no_noise,
                                        residual=residual,
                                        )
        self.norm = nn.LayerNorm(d_model)
        self.gp = gp
        self.denoise = denoise
        self.residual = residual
        self.d_model = d_model
        self.final_projection = nn.Linear(d_model, 1)
        self.enc_embedding = nn.Linear(src_input_size, d_model)
        self.dec_embedding = nn.Linear(tgt_input_size, d_model)
        self.ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_model * 4, seed=seed)

    def forward(self, enc_inputs, dec_inputs, y_true=None):

        enc_inputs = self.enc_embedding(enc_inputs)
        dec_inputs = self.dec_embedding(dec_inputs)
        mll_error = 0
        loss = 0

        enc_outputs, dec_outputs = self.forecasting_model(enc_inputs, dec_inputs)
        denoise = False if self.input_corrupt else self.denoise

        if denoise or (self.input_corrupt and self.training):

            if self.residual:

                enc_outputs_res, dec_outputs_res = self.forecasting_model(enc_outputs, dec_outputs)
                res_outputs = self.final_projection(dec_outputs_res)
                final_outputs = self.norm(self.final_projection((dec_outputs + dec_outputs_res)))

                if y_true is not None:
                    residual = y_true - res_outputs
                    loss = torch.nn.HuberLoss()(residual, res_outputs)

                return final_outputs, loss

            else:

                dec_outputs, dist = self.de_model(enc_outputs.clone(), dec_outputs.clone())
                final_outputs = self.final_projection(dec_outputs[:, -self.pred_len:, :])

        else:
            final_outputs = self.final_projection(dec_outputs)

        if y_true is not None and not self.residual:
            loss = torch.nn.HuberLoss()(final_outputs, y_true)

        return final_outputs, loss
