import torch

from pytoda.datasets.utils.background_tensors import constant_value_tensor, range_tensor
from pytoda.datasets.utils.wrappers import WrapperCDist, WrapperKLDiv

METRIC_FUNCTION_FACTORY = {'p-norm': WrapperCDist, 'KL': WrapperKLDiv}

DISTRIBUTION_FUNCTION_FACTORY = {
    'normal': torch.distributions.normal.Normal,
    'beta': torch.distributions.beta.Beta,
    'uniform': torch.distributions.uniform.Uniform,
    'bernoulli': torch.distributions.bernoulli.Bernoulli,
}

BACKGROUND_TENSOR_FACTORY = {
    'constant': constant_value_tensor,
    'range': range_tensor,
}
