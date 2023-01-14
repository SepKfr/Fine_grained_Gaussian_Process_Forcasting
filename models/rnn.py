import random

import gpytorch
import numpy as np
import torch
from torch import nn

from models.eff_acat import process_model


class RNN(nn.Module):
    def __init__(self, n_layers, hidden_size,
                 src_input_size, device,
                 d_r, seed, pred_len,
                 dae, gp):

        super(RNN, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.lstm = nn.LSTM(src_input_size, hidden_size, n_layers, dropout=d_r)
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.linear2 = nn.Linear(hidden_size, 1, bias=False)

        self.device = device
        self.pred_len = pred_len

        self.dae = dae
        if self.dae:
            self.process = process_model(gp, hidden_size, device)

    def forward(self, x_en, x_de, target=None):

        x = torch.cat((x_en, x_de), dim=1).permute(1, 0, 2)

        outputs, _ = self.lstm(x)
        outputs = outputs.transpose(0, 1)

        if self.dae:

            outputs_2, kl_loss = self.process(outputs.clone(), target)
            output = self.linear2(outputs_2)
            output = output[:, -self.pred_len:, :]
            return output, kl_loss

        else:
            output = self.linear2(outputs)
            output = output[:, -self.pred_len:, :]
            return output












