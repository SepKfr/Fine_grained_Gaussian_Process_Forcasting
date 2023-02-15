import torch.nn as nn
import torch
import random
import numpy as np
from modules.encoder import Encoder
from modules.decoder import Decoder


class Transformer(nn.Module):

    def __init__(self, src_input_size, tgt_input_size, pred_len, d_model,
                 d_ff, d_k, d_v, n_heads, n_layers, src_pad_index,
                 tgt_pad_index, device, attn_type, seed):
        super(Transformer, self).__init__()

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.attn_type = attn_type

        self.encoder = Encoder(
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=n_layers, pad_index=src_pad_index,
            device=device, attn_type=attn_type, seed=seed)
        self.decoder = Decoder(
            d_model=d_model, d_ff=d_ff,
            d_k=d_k, d_v=d_v, n_heads=n_heads,
            n_layers=1, pad_index=tgt_pad_index,
            device=device,
            attn_type=attn_type, seed=seed)

        self.enc_embedding = nn.Linear(src_input_size, d_model)
        self.dec_embedding = nn.Linear(tgt_input_size, d_model)
        self.attn_type = attn_type
        self.pred_len = pred_len
        self.device = device

    def forward(self, enc_inputs, dec_inputs):

        enc_outputs = self.enc_embedding(enc_inputs)
        dec_outputs = self.dec_embedding(dec_inputs)

        enc_outputs = self.encoder(enc_outputs)
        dec_outputs = self.decoder(dec_outputs, enc_outputs)

        return dec_outputs