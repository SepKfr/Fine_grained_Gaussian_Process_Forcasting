from functools import partial

import numpy as np
import torch
from torch import nn
import gpytorch

from models.time_grad.guassian_diffusion import cosine_beta_schedule, default


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def normal_kl(mean1, logvar1, mean2, logvar2):
    total_kl = (0.5 * (-1 + (logvar2 - logvar1) + logvar1.exp() / logvar2.exp()
                        + (mean2 - mean1).pow(2) / logvar2.exp()))

    return total_kl


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            input_size,
            beta_end=0.1,
            diff_steps=100,
            loss_type="l2",
            betas=None,
            beta_schedule="linear",
            gp=True,
    ):
        super().__init__()

        self.gp = gp
        self.denoise_fn = denoise_fn
        self.input_size = input_size
        self.model_mean_type = 'eps'
        self.denoise_fn_proj = nn.Linear(2, 1)
        self.weight = nn.Parameter(torch.randn(1), requires_grad=True)
        self.__scale = None

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        self.to_torch = partial(torch.tensor, dtype=torch.float32)

        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        if betas is not None:
            self.betas = (
                betas.detach().cpu().numpy()
                if isinstance(betas, torch.Tensor)
                else betas
            )
        else:
            if beta_schedule == "linear":
                betas = np.linspace(1e-4, beta_end, diff_steps)
            elif beta_schedule == "quad":
                betas = np.linspace(1e-4 ** 0.5, beta_end ** 0.5, diff_steps) ** 2
            elif beta_schedule == "const":
                betas = beta_end * np.ones(diff_steps)
            elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
                betas = 1.0 / np.linspace(diff_steps, 1, diff_steps)
            elif beta_schedule == "sigmoid":
                betas = np.linspace(-6, 6, diff_steps)
                betas = (beta_end - 1e-4) / (np.exp(-betas) + 1) + 1e-4
            elif beta_schedule == "cosine":
                betas = cosine_beta_schedule(diff_steps)
            else:
                raise NotImplementedError(beta_schedule)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        self.__scale = scale

    def extract(self, x, t, x_shape):

        b, *_ = t.shape
        out = x.gather(-1, t)
        out = out.reshape(b, *((1,) * (len(x_shape) - 1)))
        return out

    def q_mean_variance(self, x_start, t, gp_cov=None):

        mean = self.extract(self.sqrt_alphas_cumprod(t, x_start.shape, gp_cov), t, x_start.shape) * x_start
        variance = 1.0 - self.extract(self.alphas_cumprod(t, x_start.shape, gp_cov), t, x_start.shape)
        log_variance = torch.log(variance)
        if gp_cov is None:
            pass
        else:
            variance = self.weight * variance + (1 - self.weight) * gp_cov

        if gp_cov is None:
            pass
        else:
            log_variance = self.weight * torch.log(variance) + (1 - self.weight) * torch.log(gp_cov)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None, gp_cov=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        variance = self.weight * self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) \
                   + (1 - self.weight) * gp_cov if gp_cov is not None else  \
            self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return (
                self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + variance * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t, gp_cov=None):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self.extract(self.posterior_mean_coef1, t, x_start.shape) * x_start +
                self.extract(self.posterior_mean_coef2, t, x_start.shape) * x_t
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_start.shape)
        posterior_log_variance_clipped = self.extract(self.posterior_log_variance_clipped, t,  x_start.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_xstart_from_eps(self, x_t, t, eps, gp_cov=None):

        assert x_t.shape == eps.shape
        return (
                self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def p_mean_variance(self, denoise_fn, x, cond, t, clip_denoised: bool, return_pred_xstart: bool, gp_cov=None):

        model_output = denoise_fn(x, t, cond)

        # Learned variance

        model_output, model_log_variance = torch.chunk(model_output, 2, dim=-1)
        model_variance = torch.exp(model_log_variance)

        # Mean parameterization
        _maybe_clip = lambda x_: (x_.clamp(-1, 1) if clip_denoised else x_)
        if self.model_mean_type == 'xprev':  # the model predicts x_{t-1}
            pred_xstart = _maybe_clip(self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output, gp_cov=gp_cov))
            model_mean = model_output
        elif self.model_mean_type == 'xstart':  # the model predicts x_0
            pred_xstart = _maybe_clip(model_output)
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t, gp_cov=gp_cov)
        elif self.model_mean_type == 'eps':  # the model predicts epsilon
            pred_xstart = _maybe_clip(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output, gp_cov=gp_cov))
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t, gp_cov=gp_cov)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, pred_xstart
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_xprev(self, x_t, t, xprev, gp_cov=None):

        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
                1.0 / self.extract(self.posterior_mean_coef1(t, x_t.shape, gp_cov), t, x_t.shape) * xprev
                - self.extract(self.posterior_mean_coef2(t, x_t.shape, gp_cov), t, x_t.shape)
                / self.extract(self.posterior_mean_coef1, t, x_t.shape)
        ) * x_t

    def p_sample(self, denoise_fn, *, x, cond, t, clip_denoised=True, gp_cov=None):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, x=x, cond=cond, t=t, clip_denoised=clip_denoised, gp_cov=gp_cov, return_pred_xstart=True)
        noise = torch.randn_like(model_mean)
        assert noise.shape == x.shape
        # no noise when t == 0
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return sample, pred_xstart

    def p_sample_loop(
        self,
        model,
        shape,
        cond,
        noise=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            cond,
            noise=noise,
            device=device,
            progress=progress,
        ):
            final = sample
        return final

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        cond,
        noise=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device

        img = torch.randn(*shape, device=device)
        B, T, _ = img.shape
        if self.gp:

            mean = self.mean_module(img)
            co_var = self.covar_module(img)
            gp_cov = gpytorch.distributions.MultivariateNormal(mean, co_var).sample().unsqueeze(-1)
            gp_cov.reshape(B * T, 1, -1)
        else:
            gp_cov = None

        img = img.reshape(B*T, 1, -1)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.fill(torch.zeros((B*T, )), i).long()
            with torch.no_grad():
                sample, pred_xstart = self.p_sample(
                    denoise_fn=model, x=img,
                    cond=cond.reshape(B*T, 1, -1), t=t, gp_cov=gp_cov)
                yield sample
                img = sample

    def p_losses(self, x_start, cond, t, noise=None):

        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.gp:
            mean = self.mean_module(x_start)
            co_var = self.covar_module(x_start)
            gp_cov = gpytorch.distributions.MultivariateNormal(mean, co_var).sample().unsqueeze(-1)
            B, T, _ = gp_cov.shape
            gp_cov = gp_cov.reshape(B*T, 1, -1)
        else:
            gp_cov = None

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise, gp_cov=gp_cov)
        x_recon = self.denoise_fn(x_noisy, t, cond=cond)
        x_recon, _ = torch.chunk(x_recon, 2, dim=-1)

        return x_noisy, x_recon

    def log_prob(self, x, cond, *args, **kwargs):

        B, T, _ = x.shape

        time = torch.randint(0, self.num_timesteps, (B * T,), device=x.device).long()
        x_noisy, x_rec = self.p_losses(
            x.reshape(B*T, 1, -1), cond.reshape(B*T, 1, -1), time, *args, **kwargs
        )

        return x_noisy, x_rec


