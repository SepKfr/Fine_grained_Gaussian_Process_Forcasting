import torch
import torch.nn as nn
import numpy as np
import random


class ATA(nn.Module):
    def __init__(self, d_k, device, h, seed):

        super(ATA, self).__init__()

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.d_k = d_k
        self.filter_length = [3, 9]

        self.conv_list_k = nn.ModuleList([
            nn.Sequential(nn.Conv1d(
                in_channels=d_k*h, out_channels=d_k*h, kernel_size=f, padding=int((f-1)/2), device=device),
                          nn.BatchNorm1d(d_k*h, device=device),
                          nn.ReLU())
            for f in self.filter_length
            ])

        self.conv_list_q = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=d_k*h, out_channels=d_k*h, kernel_size=f, padding=int((f-1)/2), device=device),
                          nn.BatchNorm1d(d_k*h, device=device),
                          nn.ReLU())
            for f in self.filter_length]).to(device)


    def forward(self, Q, K, V):

        b, h, l, d_k = Q.shape
        l_k = K.shape[2]
        Q_l = []
        K_l = []

        Q = Q.reshape(b, -1, l)
        K = K.reshape(b, -1, l_k)

        [Q_l.append(self.conv_list_q[i](Q)) for i in range(len(self.filter_length))]
        [K_l.append(self.conv_list_k[i](K)) for i in range(len(self.filter_length))]

        Q_p = torch.cat(Q_l, dim=0).reshape(b, h, l * len(self.filter_length), -1)
        K_p = torch.cat(K_l, dim=0).reshape(b, h, l_k * len(self.filter_length), -1)

        Q, _ = torch.topk(Q_p, dim=-2, k=l)

        K, _ = torch.topk(K_p, dim=-2, k=l_k)

        scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(self.d_k)

        attn = torch.softmax(scores, -1)
        context = torch.einsum('bhqk,bhkd->bhqd', attn, V)
        return context, attn