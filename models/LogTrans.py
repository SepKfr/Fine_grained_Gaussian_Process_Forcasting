import math
import random
import numpy as np
import torch
import torch.nn as nn


class LogTrans(nn.Module):
    def __init__(self, d_k, device, seed):

        super(LogTrans, self).__init__()

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.device = device
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):

        l_k = K.shape[2]
        log_l_k = int(math.log2(l_k))
        inds = [l_k - 2 ** (log_l_k - i) for i in range(0, log_l_k)]
        K = K[:, :, inds, :]
        scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(self.d_k)

        if attn_mask is not None:
            attn_mask = attn_mask[:, :, :, inds]
            attn_mask = torch.as_tensor(attn_mask, dtype=torch.bool)
            attn_mask = attn_mask.to(self.device)
            scores.masked_fill_(attn_mask, -1e9)

        attn = torch.softmax(scores, -1)
        V = V[:, :, inds, :]
        context = torch.einsum('bhqk,bhvd->bhqd', attn, V)
        return context, None, attn