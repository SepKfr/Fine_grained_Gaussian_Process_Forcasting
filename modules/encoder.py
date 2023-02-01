import random

import numpy as np
import torch
import torch.nn as nn
from modules.multi_head_attention import MultiHeadAttention
from modules.feedforward import PoswiseFeedForwardNet
from modules.encoding import PositionalEncoding


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads,
                 device, attn_type, seed):
        super(EncoderLayer, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.enc_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device,
            attn_type=attn_type, seed=seed)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff, seed=seed)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, enc_inputs, enc_self_attn_mask=None):

        out, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, attn_mask=enc_self_attn_mask)
        out = self.layer_norm(out + enc_inputs)
        out_2 = self.pos_ffn(out)
        out_2 = self.layer_norm(out_2 + out)
        return out_2


class Encoder(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v, n_heads,
                 n_layers, pad_index, device,
                 attn_type, seed):
        super(Encoder, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.device = device
        self.pad_index = pad_index
        self.attn_type = attn_type
        self.pos_emb = PositionalEncoding(
            d_hid=d_model,
            device=device)
        self.n_layers = n_layers
        self.layers = []
        for _ in range(n_layers):
            encoder_layer = EncoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v, n_heads=n_heads,
                device=device,
                attn_type=attn_type, seed=seed)
            self.layers.append(encoder_layer)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, enc_input):

        enc_outputs = self.pos_emb(enc_input)

        enc_self_attn_mask = None

        for layer in self.layers:
            enc_outputs = layer(enc_outputs, enc_self_attn_mask)

        return enc_outputs