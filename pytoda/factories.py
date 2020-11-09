import torch.distributions as d
from .wrappers import WrapperCDist, WrapperEMD, WrapperKLDiv

METRIC_FUNCTION_FACTORY = {
    'p-norm': WrapperCDist,
    'wasserstein': WrapperEMD,
    'KL': WrapperKLDiv
}

DISTRIBUTION_FUNCTION_FACTORY = {
    'normal': d.normal.Normal,
    'beta': d.beta.Beta,
    'uniform': d.uniform.Uniform,
    'bernoulli': d.bernoulli.Bernoulli
}
