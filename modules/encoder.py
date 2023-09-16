import random
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Dropout
from modules.multi_head_attention import MultiHeadAttention
from modules.feedforward import PoswiseFeedForwardNet


class EncoderLayer(nn.Module):
    def __init__(self, attn_type, d_k, d_v, d_model, d_ff, n_heads, device, seed, dropout=0.0):
        """
        Encoder Layer in the Transformer model.

        :param d_model: Dimensionality of the model.
        :param d_ff: Dimensionality of the feedforward layer.
        :param n_heads: Number of attention heads.
        :param device: Device on which the model is executed.
        :param seed: Random seed for reproducibility.
        :param dropout: Dropout rate.
        """
        super(EncoderLayer, self).__init__()

        factory_kwargs = {'device': device}

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.enc_self_attn = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, device=device, seed=seed, attn_type=attn_type, d_k=d_k, d_v=d_v)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff, seed=seed)

        self.norm1 = nn.LayerNorm(d_model, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, **factory_kwargs)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.enc_self_attn(x, x, x))
        x = self.norm2(x + self.pos_ffn(x))
        return x


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, seed):
        """
        Encoder in the Transformer model.

        :param encoder_layer: Instance of EncoderLayer.
        :param num_layers: Number of encoder layers.
        :param seed: Random seed for reproducibility.
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, enc_input):
        enc_outputs = enc_input

        for layer in self.layers:
            enc_outputs = layer(enc_outputs)

        return enc_outputs
