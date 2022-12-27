import torch
from torch import nn
import torch.nn.functional as F

from models.ACAT_diffusion.guassian_diffusion import GaussianDiffusion
from models.eff_acat import Transformer
from models.time_grad.epsilon_theta import EpsilonTheta
from models.eff_acat import PoswiseFeedForwardNet


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
                 diff_steps=50,
                 loss_type="l2",
                 beta_end=0.1,
                 beta_schedule="cosine",
                 attn_type="KittyCat",
                 gp_cov=False):
        super(ACATTrainingNetwork, self).__init__()

        self.gp_cov = gp_cov
        self.device = device
        self.target_dim = d_model
        self.loss_type = loss_type
        self.pred_len = pred_len

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
            seed=seed,
        )

        self.diffusion = GaussianDiffusion(
            self.denoise_fn,
            input_size=tgt_input_size,
            diff_steps=diff_steps,
            loss_type=loss_type,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            gp=self.gp_cov,
            seed=seed
        )

    def forward(self, x_en, x_de, target):

        B = x_de.shape[0]

        model_output = self.model(x_en, x_de)

        x_recon, noise, sample = self.diffusion.log_prob(model_output)

        output = sample.reshape(B, self.pred_len, -1)

        loss = nn.MSELoss()(output, target)

        return loss

    def predict(self, x_en, x_de):

        B = x_de.shape[0]
        model_output = self.model(x_en, x_de)
        _, _, samples = self.diffusion.log_prob(model_output)
        new_samples = samples.reshape(B, self.pred_len, -1)

        return new_samples
