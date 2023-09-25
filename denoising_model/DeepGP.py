import random

import gpytorch
import numpy as np
import torch
import torch.nn as nn
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.variational import VariationalStrategy, MeanFieldVariationalDistribution, CholeskyVariationalDistribution


class ToyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, nu, input_dims, output_dims, seed, num_inducing, mean_type='constant'):

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)

        kernel = RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims)
        self.covar_module = ScaleKernel(
            kernel,
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class DeepGPp(DeepGP):
    def __init__(self, nu, num_hidden_dims, seed):

        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=num_hidden_dims,
            output_dims=None,
            mean_type='linear',
            seed=seed,
            nu=nu,
            num_inducing=128
        )

        super().__init__()
        self.num_hidden_dims = num_hidden_dims
        self.hidden_layer = hidden_layer
        self.embedding = nn.Linear(1, num_hidden_dims)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, inputs):

        dist = self.hidden_layer(inputs)
        return dist

    def predict(self, x):

        dist = self(x)
        preds = self.likelihood(dist).to_data_independent_dist()
        mean = self.embedding(preds.mean.mean(0).unsqueeze(-1)) * 0.05
        return dist, mean