import random
import numpy as np
import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, n_layers, hidden_size,
                 src_input_size, tgt_input_size,
                 rnn_type, device, d_r, seed, pred_len):

        super(RNN, self).__init__()

        self.lstm = nn.LSTM(src_input_size, hidden_size, n_layers, dropout=d_r)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.linear2 = nn.Linear(hidden_size, 1, bias=False)
        self.hidden = None
        self.device = device
        self.pred_len = pred_len
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def forward(self, x_en, x_de):

        x = torch.cat((x_en, x_de), dim=1).permute(1, 0, 2)

        if self.hidden is None:
            self.hidden = torch.zeros(self.n_layers, x.shape[1], self.hidden_size).to(self.device)

        outputs, _ = self.lstm(x, (self.hidden, self.hidden))

        outputs = self.linear2(outputs).transpose(0, 1)
        outputs = outputs[:, -self.pred_len:, :]

        return outputs