import random
import numpy as np
import torch
import torch.nn as nn
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO

from denoising_model.DeepGP import DeepGPp
from denoising_model.denoise_model_2 import denoise_model_2
from forecasting_models.LSTM import RNN
from modules.transformer import Transformer
torch.autograd.set_detect_anomaly(True)


class Forecast_denoising(nn.Module):
    def __init__(self, model_name:str, config: tuple, gp: bool,
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
        self.gp = gp
        self.lam = nn.Parameter(torch.randn(1))

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

        self.d = d_model
        self.de_model = denoise_model_2(self.forecasting_model,
                                        model_name, gp,
                                        d_model, device, seed,
                                        n_noise=no_noise,
                                        residual=residual)
        self.denoise = denoise
        self.residual = residual
        self.final_projection = nn.Linear(d_model, 1)
        self.enc_embedding = nn.Linear(src_input_size, d_model)
        self.dec_embedding = nn.Linear(tgt_input_size, d_model)
        self.deep_gp = DeepGPp(d_model, seed)

    def forward(self, enc_inputs, dec_inputs, y_true=None):

        mll_error = 0
        loss = 0
        mse_loss = 0

        enc_inputs = self.enc_embedding(enc_inputs)
        dec_inputs = self.dec_embedding(dec_inputs)

        enc_outputs, dec_outputs = self.forecasting_model(enc_inputs, dec_inputs)
        forecasting_model_outputs = self.final_projection(dec_outputs[:, -self.pred_len:, :])

        if self.denoise or (self.input_corrupt and self.training):

            de_model_outputs, dist = self.de_model(enc_outputs.clone(), dec_outputs.clone())
            final_outputs = self.final_projection(de_model_outputs[:, -self.pred_len:, :])

            if self.gp and self.training:
                mll = DeepApproximateMLL(
                    VariationalELBO(self.de_model.deep_gp.likelihood, self.de_model.deep_gp, self.d))
                mll_error = -mll(dist, y_true.permute(2, 0, 1)).mean()

            if self.residual:

                enc_outputs_res, dec_outputs_res = self.forecasting_model(enc_outputs, dec_outputs)
                res_outputs = self.final_projection(dec_outputs_res[:, -self.pred_len:, :])
                final_outputs = forecasting_model_outputs + res_outputs
                if y_true is not None:
                    residual = y_true - forecasting_model_outputs
                    loss = nn.MSELoss()(y_true, residual)
        else:
            final_outputs = forecasting_model_outputs

        if y_true is not None:
            mse_loss = nn.MSELoss()(y_true, final_outputs)
            if self.training:
                loss = mse_loss + torch.clip(self.lam, min=0, max=0.01) * mll_error
            else:
                loss = mse_loss
        return final_outputs, loss, mse_loss
