import random
import numpy as np
import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, n_layers, hidden_size,
                 src_input_size,
                 seed):

        super(RNN, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.lstm = nn.LSTM(src_input_size, hidden_size, n_layers)
        self.proj_back = nn.Linear(hidden_size, 1)
        self.n_layers = n_layers
        self.hidden_size = hidden_size

    def forward(self, x):

        outputs, _ = self.lstm(x)
        return outputs













