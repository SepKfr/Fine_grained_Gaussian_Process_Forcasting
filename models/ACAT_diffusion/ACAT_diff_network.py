import torch
from torch import nn
import torch.nn.functional as F

from models.ACAT_diffusion.guassian_diffusion import GaussianDiffusion
from models.eff_acat import Transformer
from models.time_grad.epsilon_theta import EpsilonTheta


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
                 conditioning_length,
                 residual_layers=8,
                 residual_channels=8,
                 dilation_cycle_length=2,
                 diff_steps=100,
                 loss_type="l2",
                 beta_end=0.1,
                 beta_schedule="linear",
                 attn_type="KittyCat",
                 gp=True
                 , ):
        super(ACATTrainingNetwork, self).__init__()

        self.loss_type = loss_type
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

        self.denoise_fn = EpsilonTheta(
            target_dim=1,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
        )

        self.diffusion = GaussianDiffusion(
            self.denoise_fn,
            input_size=tgt_input_size,
            diff_steps=diff_steps,
            loss_type=loss_type,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            gp=gp
        )

    def forward(self, x_en, x_de, target):

        model_output = self.model(x_en, x_de)

        x_noisy, x_rec = self.diffusion.log_prob(target, model_output)

        if self.loss_type == "l1":
            loss = F.l1_loss(x_rec, x_noisy)
        elif self.loss_type == "l2":
            loss = F.mse_loss(x_rec, x_noisy)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(x_rec, x_noisy)
        else:
            raise NotImplementedError()

        return loss.mean()

    def predict(self, x_en, x_de):

        model_output = self.model(x_en, x_de)
        new_samples = self.diffusion.p_sample_loop(cond=model_output, model=self.denoise_fn, shape=model_output.shape)

        return new_samples




