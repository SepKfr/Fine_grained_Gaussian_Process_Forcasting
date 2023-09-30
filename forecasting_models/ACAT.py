import random
import numpy as np
import torch
import torch.nn as nn


class ACAT(nn.Module):

    def __init__(self, d_k, device, h, seed):

        super(ACAT, self).__init__()

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.device = device
        self.d_k = d_k
        self.filter_length = [1, 3, 7, 9]
        self.conv_list_q = nn.ModuleList(
            [nn.Conv1d(in_channels=d_k*h, out_channels=d_k*h,
                       kernel_size=f,
                       padding=int(f/2),
                       bias=False) for f in self.filter_length]).to(device)
        self.conv_list_k = nn.ModuleList(
            [nn.Conv1d(in_channels=d_k*h, out_channels=d_k*h,
                       kernel_size=f,
                       padding=int(f/2),
                       bias=False) for f in self.filter_length]).to(device)
        self.norm = nn.BatchNorm1d(h * d_k).to(device)
        self.activation = nn.ELU().to(device)

    def forward(self, Q, K, V):

        b, h, l, d_k = Q.shape
        l_k = K.shape[2]

        len_n_k = len(self.filter_length)

        Q_l = [self.activation(self.norm(self.conv_list_q[i](Q.reshape(b, h*d_k, l))))[:, :, :l]
               for i in range(len(self.filter_length))]
        K_l = [self.activation(self.norm(self.conv_list_k[i](K.reshape(b, h * d_k, l_k))))[:, :, :l_k]
               for i in range(len(self.filter_length))]
        Q_p = torch.cat(Q_l, dim=0).reshape(b, h, len_n_k, l, d_k)
        K_tmp = torch.cat(K_l, dim=0).reshape(b, h, len_n_k, l_k, d_k)

        m_f = max(self.filter_length)
        K_p = K_tmp[:, :, :, 0::m_f, :]

        scores = torch.einsum('bhpqd,bhpkd->bhpqk', Q_p, K_p) / np.sqrt(self.d_k)

        attn = torch.softmax(scores, -1)
        attn, _ = torch.max(attn, dim=2)
        attn_f = torch.zeros(b, h, l, l_k).to(self.device)
        attn_f[:, :, :, 0::m_f] = attn
        attn_f = torch.softmax(attn_f, -1)
        context = torch.einsum('bhqk,bhkd->bhqd', attn_f, V)
        return context, attn_f