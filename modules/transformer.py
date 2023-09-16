import torch.nn as nn
import torch
import random
import numpy as np
from modules.encoder import Encoder, EncoderLayer
from modules.decoder import Decoder, DecoderLayer


class Transformer(nn.Module):

    def __init__(self, pred_len, d_model,
                 d_ff, d_k, d_v, n_heads, n_layers,
                 device, attn_type, seed):
        super(Transformer, self).__init__()

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.attn_type = attn_type

        self.encoder = Encoder(EncoderLayer(d_ff=d_ff,
                                            d_model=d_model,
                                            n_heads=n_heads,
                                            device=device,
                                            seed=seed,
                                            attn_type=attn_type,
                                            d_k=d_k,
                                            d_v=d_v),
                               n_layers,
                               seed)
        self.decoder = Decoder(DecoderLayer(d_ff=d_ff,
                                            d_model=d_model,
                                            n_heads=n_heads,
                                            device=device,
                                            seed=seed,
                                            attn_type=attn_type,
                                            d_k=d_k,
                                            d_v=d_v
                                            ),
                               n_layers,
                               seed)

        self.attn_type = attn_type
        self.pred_len = pred_len
        self.device = device

    def forward(self, enc_inputs, dec_inputs):

        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs, enc_outputs)

        return enc_outputs, dec_outputs