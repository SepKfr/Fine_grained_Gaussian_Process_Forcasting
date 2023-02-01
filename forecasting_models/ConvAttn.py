import numpy as np
import torch
import torch.nn as nn
import random

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class ConvAttn(nn.Module):

    def __init__(self, d_k, h, kernel, device, seed):

        super(ConvAttn, self).__init__()

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.device = device
        self.d_k = d_k
        self.conv_q = nn.Conv1d(in_channels=d_k*h, out_channels=d_k*h,
                                kernel_size=kernel,
                                padding=int(kernel/2), bias=False).to(device)
        self.conv_k = nn.Conv1d(in_channels=d_k * h, out_channels=d_k * h,
                                kernel_size=kernel,
                                padding=int(kernel / 2), bias=False).to(device)

    def forward(self, Q, K, V, attn_mask):

        b, h, l, d_k = Q.shape
        l_k = K.shape[2]

        Q = self.conv_q(Q.reshape(b, h*d_k, l))[:, :, :l].reshape(b, h, l, d_k)
        K = self.conv_k(K.reshape(b, h*d_k, l_k))[:, :, :l_k].reshape(b, h, l_k, d_k)

        scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(self.d_k)
        attn = torch.softmax(scores, -1)
        context = torch.einsum('bhqk,bhvd->bhqd', attn, V)
        return context, attn