import torch.nn as nn
import torch


class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, d_hid, device, max_len=1000):

        super(PositionalEncoding, self).__init__()
        # Create a long enough `P`
        self.P = torch.zeros((1, max_len, d_hid)).to(device)
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, d_hid, 2, dtype=torch.float32) / d_hid)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return X
