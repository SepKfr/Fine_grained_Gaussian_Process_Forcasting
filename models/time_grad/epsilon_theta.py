import math

import torch
from torch import nn
import torch.nn.functional as F
from models.eff_acat import PositionalEncoding


class DiffusionEmbedding(nn.Module):
    def __init__(self, dim, proj_dim, max_steps=500):
        super().__init__()

        self.embedding = PositionalEncoding(dim, max_steps)
        self.projection1 = nn.Linear(dim, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):

        x = self.embedding(diffusion_step)
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, dim, max_steps):

        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Conv1d(
            1, 2 * residual_channels, 1, padding_mode="circular"
        )
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, conditioner, diffusion_step):

        diffusion_step = self.diffusion_projection(diffusion_step)
        conditioner = self.conditioner_projection(conditioner.permute(0, 2, 1)).permute(0, 2, 1)

        y = x + diffusion_step
        y = self.dilated_conv(y.permute(0, 2, 1)).permute(0, 2, 1) + conditioner

        gate, filter = torch.chunk(y, 2, dim=-1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=-1)
        return (x + residual) / math.sqrt(2.0), skip


class CondUpsampler(nn.Module):
    def __init__(self, target_dim, res_dim):
        super().__init__()
        self.linear1 = nn.Linear(target_dim, res_dim)

    def forward(self, x):

        x = self.linear1(x)
        x = F.leaky_relu(x, 0.4)

        return x


class EpsilonTheta(nn.Module):
    def __init__(
        self,
        target_dim=1,
        time_emb_dim=16,
        residual_layers=8,
        residual_channels=8,
        dilation_cycle_length=2,
        residual_hidden=64,
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(
            1, residual_channels, 1
        )
        self.diffusion_embedding = DiffusionEmbedding(
            time_emb_dim, proj_dim=residual_hidden
        )
        self.cond_upsampler = CondUpsampler(
            target_dim=target_dim, res_dim=residual_hidden
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    hidden_size=residual_hidden,
                )
                for i in range(residual_layers)
            ]
        )
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 3, padding=int((3-1)/2))
        self.output_projection = nn.Conv1d(residual_channels, 2, 1)

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, inputs, time, cond):

        time = time.unsqueeze(-1).repeat(1, inputs.shape[1]).unsqueeze(-1)
        x = self.input_projection(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x, 0.4)

        diffusion_step = self.diffusion_embedding(time)
        #cond_up = self.cond_upsampler(cond)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x