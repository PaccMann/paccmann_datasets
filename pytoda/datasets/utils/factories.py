import torch
from pytoda.datasets.utils.wrappers import (
    WrapperCDist,
    WrapperKLDiv,
    WrapperBackgroundTensor,
)

METRIC_FUNCTION_FACTORY = {'p-norm': WrapperCDist, 'KL': WrapperKLDiv}

DISTRIBUTION_FUNCTION_FACTORY = {
    'normal': torch.distributions.normal.Normal,
    'beta': torch.distributions.beta.Beta,
    'uniform': torch.distributions.uniform.Uniform,
    'bernoulli': torch.distributions.bernoulli.Bernoulli,
}

BACKGROUND_TENSOR_FACTORY = {
    'constant': WrapperBackgroundTensor.constant_value_tensor,
    'range': WrapperBackgroundTensor.range_tensor,
}