import random
import numpy as np
import torch
from torch import nn


class DeepAr(nn.Module):
    def __init__(self, n_layers, hidden_size,
                 src_input_size, device, d_r, seed, pred_len):

        super(DeepAr, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.lstm = nn.LSTM(src_input_size, hidden_size, n_layers, dropout=d_r)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.device = device
        self.pred_len = pred_len

        self.distribution_mu = nn.Linear(hidden_size, 1)
        self.distribution_presigma = nn.Linear(hidden_size, 1)
        self.distribution_sigma = nn.Softplus()
        self.proj_out = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):

        x = x.permute(1, 0, 2)

        if hidden is None:

            hidden = torch.zeros(self.n_layers, x.shape[1], self.hidden_size).to(self.device)

        outputs, (hidden, hidden) = self.lstm(x, (hidden, hidden))

        hidden = hidden.transpose(0, 1)

        mu = self.distribution_mu(hidden)
        sigma = self.distribution_sigma(self.distribution_presigma(hidden))
        hidden = self.proj_out(hidden)

        return hidden, mu, sigma