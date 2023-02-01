import random

import numpy as np
import torch
import torch.nn as nn
from forecasting_models.ATA import ATA
from forecasting_models.ConvAttn import ConvAttn
from forecasting_models.Informer import ProbAttention
from forecasting_models.Autoformer import AutoCorrelation


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, n_heads, device, attn_type, seed):

        super(MultiHeadAttention, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.WQ = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.WK = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.WV = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.device = device

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_type = attn_type
        self.seed = seed

    def forward(self, Q, K, V, attn_mask):

        batch_size = Q.shape[0]
        q_s = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # ATA forecasting model

        if self.attn_type == "ATA":
            context, attn = ATA(d_k=self.d_k, device=self.device, h=self.n_heads, seed=self.seed)(
            Q=q_s, K=k_s, V=v_s, attn_mask=attn_mask)

        # Autoformer forecasting model

        elif self.attn_type == "autoformer":
            context, attn = AutoCorrelation(seed=self.seed)(q_s.transpose(1, 2), k_s.transpose(1, 2), v_s.transpose(1, 2),
                                              attn_mask)

        # CNN-trans forecasting model

        elif self.attn_type == "conv_attn":
            context, attn = ConvAttn(d_k=self.d_k, device=self.device, seed=self.seed, kernel=9, h=self.n_heads)(
            Q=q_s, K=k_s, V=v_s, attn_mask=attn_mask)

        # Informer forecasting model

        elif self.attn_type == "informer":
            mask_flag = True if attn_mask is not None else False
            context, attn = ProbAttention(mask_flag=mask_flag, seed=self.seed)(q_s, k_s, v_s, attn_mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output, attn