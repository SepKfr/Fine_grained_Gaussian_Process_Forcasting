import random
import numpy as np
import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, n_layers, hidden_size,
                 src_input_size, tgt_input_size,
                 rnn_type, device, d_r, seed,
                 pred_len, p_model):

        super(RNN, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.lstm = nn.LSTM(src_input_size, hidden_size, n_layers, dropout=d_r)
        self.musig = nn.Linear(hidden_size, hidden_size*2)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.linear2 = nn.Linear(hidden_size, 1, bias=False)
        self.hidden = None
        self.device = device
        self.p_model = p_model
        self.pred_len = pred_len

    def forward(self, x_en, x_de):

        x = torch.cat((x_en, x_de), dim=1).permute(1, 0, 2)

        if self.hidden is None:
            self.hidden = torch.zeros(self.n_layers, x.shape[1], self.hidden_size).to(self.device)

        outputs, _ = self.lstm(x, (self.hidden, self.hidden))

        if self.p_model:

            musig = self.musig(outputs)
            mu, sigma = musig[:, :, :self.hidden_size], musig[:, :, -self.hidden_size:]
            z = mu + torch.exp(sigma * 0.5) * torch.randn_like(sigma, device=self.device)
            output = z
            mu = torch.flatten(mu, start_dim=1)
            sigma = torch.flatten(mu, start_dim=1)
        else:
            output = outputs

        output = self.linear2(output).transpose(0, 1)
        output = output[:, -self.pred_len:, :]

        if self.p_model:
            return output, mu, sigma
        else:
            return output