import torch
import torch.nn as nn
import numpy as np
import random

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class ATA(nn.Module):
    def __init__(self, d_k, device, h, seed):

        super(ATA, self).__init__()

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.device = device
        self.d_k = d_k
        self.filter_length = [3, 9]

        self.proj_q = nn.Linear(d_k, 1, bias=False, device=device)
        self.proj_k = nn.Linear(d_k, 1, bias=False, device=device)

        self.conv_list_k = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=h, out_channels=h, kernel_size=f, padding=int((f-1)/2)),
                          nn.BatchNorm1d(h),
                          nn.Softmax(dim=-1))
            for f in self.filter_length
            ]).to(device)

        self.conv_list_q = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=h, out_channels=h, kernel_size=f, padding=int((f-1)/2)),
                          nn.BatchNorm1d(h),
                          nn.Softmax(dim=-1)
                          )
            for f in self.filter_length]).to(device)

        self.proj_back_q = nn.Linear(1, self.d_k, bias=False).to(device)
        self.proj_back_k = nn.Linear(1, self.d_k, bias=False).to(device)

        self.factor = 1

    def forward(self, Q, K, V, attn_mask):

        b, h, l, d_k = Q.shape
        l_k = K.shape[2]
        Q_l = []
        K_l = []

        Q = self.proj_q(Q)
        K = self.proj_k(K)

        Q = Q.reshape(b, -1, l)
        K = K.reshape(b, -1, l_k)

        [Q_l.append(self.conv_list_q[i](Q)) for i in range(len(self.filter_length))]
        [K_l.append(self.conv_list_k[i](K)) for i in range(len(self.filter_length))]

        Q_p = torch.cat(Q_l, dim=0).reshape(b, h, l * len(self.filter_length), -1)
        K_p = torch.cat(K_l, dim=0).reshape(b, h, l_k * len(self.filter_length), -1)

        Q_proj = Q_p.reshape(b, h, l, -1)
        Q = torch.max(Q_proj, dim=-1)[0].unsqueeze(-1)
        Q = self.proj_back_q(Q)

        K_proj = K_p.reshape(b, h, l_k, -1)
        K = torch.max(K_proj, dim=-1)[0].unsqueeze(-1)
        K = self.proj_back_k(K)

        scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(self.d_k)

        attn = torch.softmax(scores, -1)
        context = torch.einsum('bhqk,bhkd->bhqd', attn, V)
        return context, attn