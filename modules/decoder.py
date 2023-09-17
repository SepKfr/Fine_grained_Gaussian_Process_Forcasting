import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Dropout
from modules.multi_head_attention import MultiHeadAttention
from modules.feedforward import PoswiseFeedForwardNet
import copy


class DecoderLayer(nn.Module):
    def __init__(self, attn_type, d_model, d_k, d_v, d_ff, n_heads, device, seed, dropout=0.0):
        """
        Decoder Layer in the Transformer model.

        :param d_model: Dimensionality of the model.
        :param d_ff: Dimensionality of the feedforward layer.
        :param n_heads: Number of attention heads.
        :param device: Device on which the model is executed.
        :param seed: Random seed for reproducibility.
        :param dropout: Dropout rate.
        """
        factory_kwargs = {'device': device}

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, device=device, seed=seed, attn_type=attn_type, d_k=d_k, d_v=d_v)
        self.dec_enc_attn = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, device=device, seed=seed, attn_type=attn_type, d_k=d_k, d_v=d_v)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff, seed=seed)

        self.norm1 = nn.LayerNorm(d_model, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x, memory):

        x1 = self.norm1(x + self.self_attention(x, x, x))
        x2 = self.norm2(x1 + self.dec_enc_attn(x1, memory, memory))
        x3 = self.norm3(x2 + self.pos_ffn(x2))
        return x3


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, seed):
        """
        Decoder in the Transformer model.

        :param decoder_layer: Instance of DecoderLayer.
        :param num_layers: Number of decoder layers.
        :param seed: Random seed for reproducibility.
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        super(Decoder, self).__init__()

        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(self, dec_inputs, enc_outputs):
        dec_outputs = dec_inputs

        for module in self.layers:
            dec_outputs = module(
                x=dec_outputs,
                memory=enc_outputs
            )
        return dec_outputs
