import random
import numpy as np
import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, n_layers, hidden_size,
                 src_input_size, device,
                 d_r, seed, pred_len):

        super(RNN, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.enc_lstm = nn.LSTM(src_input_size, hidden_size, n_layers, dropout=d_r)
        self.dec_lstm = nn.LSTM(src_input_size, hidden_size, n_layers, dropout=d_r)
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.device = device
        self.pred_len = pred_len

    def forward(self, x_en, x_de):

        enc_outputs, _ = self.lstm(x_en)
        enc_outputs = enc_outputs.transpose(0, 1)
        dec_outputs, _ = self.lstm(x_de)
        dec_outputs = enc_outputs.transpose(0, 1)

        return enc_outputs, dec_outputs













