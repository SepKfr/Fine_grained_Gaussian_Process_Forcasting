import torch
import torch.nn as nn
from denoising_model.denoising_model import denoise_model
from forecasting_models.LSTM import RNN
from modules.transformer import Transformer


class Forecast_denoising(nn.Module):
    def __init__(self, model_name:str, config: tuple, gp: bool,
                 denoise: bool, device: torch.device,
                 seed: int, pred_len: int, attn_type: str):

        super(Forecast_denoising, self).__init__()

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
                                                 seed=seed)

        self.de_model = denoise_model(gp, d_model, device)
        self.denoise = denoise
        self.final_projection = nn.Linear(d_model, 1)

    def forward(self, enc_inputs, dec_inputs, target=None):

        outputs = self.forecasting_model(enc_inputs, dec_inputs)

        if self.denoise:

            outputs, kl_loss = self.de_model(outputs.clone(), target)
            outputs = self.final_projection(outputs[:, -self.pred_len:, :])
            return outputs, kl_loss

        else:

            outputs = self.final_projection(outputs[:, -self.pred_len:, :])
            return outputs