import torch
from torch import nn
import torch.nn.functional as F

from models.ACAT_diffusion.guassian_diffusion import GaussianDiffusion
from models.eff_acat import Transformer
from models.time_grad.epsilon_theta import UNetModel


class ACATTrainingNetwork(nn.Module):
    def __init__(self,
                 src_input_size,
                 tgt_input_size,
                 d_model,
                 pred_len,
                 d_k,
                 n_heads,
                 stack_size,
                 device,
                 seed,
                 diff_steps=500,
                 loss_type="l2",
                 beta_end=0.1,
                 beta_schedule="linear",
                 attn_type="KittyCat"
                 , ):
        super(ACATTrainingNetwork, self).__init__()

        self.target_dim = d_model
        self.loss_type = loss_type
        self.pred_len = pred_len
        self.target_proj = nn.Linear(1, self.target_dim)
        self.target_proj_back = nn.Linear(self.target_dim, 1)

        self.model = Transformer(src_input_size=src_input_size,
                                 tgt_input_size=tgt_input_size,
                                 pred_len=pred_len,
                                 d_model=d_model,
                                 d_ff=d_model * 4,
                                 d_k=d_k, d_v=d_k, n_heads=n_heads,
                                 n_layers=stack_size, src_pad_index=0,
                                 tgt_pad_index=0, device=device,
                                 attn_type=attn_type,
                                 seed=seed, kernel=1)

        self.denoise_fn = UNetModel(
            in_channels=1,
            model_channels=16,
            out_channels=1,
            num_res_blocks=1,
            attention_resolutions=(1,)
        )

        self.diffusion = GaussianDiffusion(
            self.denoise_fn,
            input_size=tgt_input_size,
            diff_steps=diff_steps,
            loss_type=loss_type,
            beta_end=beta_end,
            beta_schedule=beta_schedule
        )
        self.layer_norm = nn.LayerNorm(1, elementwise_affine=False)

    def forward(self, x_en, x_de, target):

        B = x_de.shape[0]

        model_output = self.model(x_en, x_de)

        x_recon, noise, sample = self.diffusion.log_prob(model_output)

        output = self.layer_norm(model_output + sample.reshape(B, self.pred_len, -1))

        loss = nn.MSELoss()(x_recon, noise) * 0.001 + nn.MSELoss()(output, target)

        return loss

    def predict(self, x_en, x_de):

        B = x_de.shape[0]
        model_output = self.model(x_en, x_de)
        _, _, samples = self.diffusion.log_prob(model_output)
        new_samples = samples.reshape(B, self.pred_len, -1)

        output = self.layer_norm(new_samples + model_output)

        return output
