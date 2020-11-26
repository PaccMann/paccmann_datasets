from dataclasses import dataclass
from typing import Optional

import torch
from torch.distributions.distribution import Distribution
from torch.random import fork_rng
from torch.utils.data import Dataset

from pytoda.datasets.utils.factories import DISTRIBUTION_FUNCTION_FACTORY
from pytoda.types import Any, Tensor


@dataclass
class StochasticItems:
    """Sample an item from a distribution on the fly on indexing.

    Args:
        distribution (Distribution): the distribution to sample from.
        shape (torch.Size): the desired shape of each item.
        device (torch.device): device to send the tensor to.
    """

    distribution: Distribution
    shape: torch.Size
    device: torch.device

    def __getitem__(self, index: Any) -> Tensor:
        """Samples an item.

        Args:
            index (Any): is ignored.

        Returns:
            Tensor: sampled from distribution with given shape.
        """
        return self.distribution.sample(self.shape).to(self.device)


class SyntheticDataset(Dataset):
    """Generates 2D samples from a specified distribution."""

    def __init__(
        self,
        dataset_size: int,
        dataset_dim: int,
        dataset_depth: int = 1,
        distribution_type: str = 'normal',
        distribution_args: dict = {'loc': 0.0, 'scale': 1.0},
        seed: Optional[int] = None,
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
    ) -> None:
        """Dataset of synthetic 2D samples from a specified distribution

        Args:
            dataset_size (int): Number of samples to generate (N).
            dataset_dim (int): Feature size / dimension of each sample (H).
            dataset_depth (int): Length of time series per sample (T). This is to
                support 2D samples. Sampling from __getitem__ will have shape
                T x H. T defaults to 1.
            distribution_type (str): The distribution from which data should
                be sampled. Defaults to 'normal'. For full list see:
                ``pytoda.utils.factories.DISTRIBUTION_FUNCTION_FACTORY``
            distribution_args (dict): dictionary of keyword arguments for
                the distribution specified above.
            seed (Optional[int]): If passed, all samples are generated once with
                this seed (using a local RNG only). Defaults to None, where
                individual samples are generated when the SyntheticDataset is
                indexed (using the global RNG).
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
        """

        super(SyntheticDataset, self).__init__()

        if distribution_type not in DISTRIBUTION_FUNCTION_FACTORY.keys():
            raise KeyError(
                f'distribution_type was {distribution_type}, should be from '
                f'{DISTRIBUTION_FUNCTION_FACTORY.keys()}.'
            )

        self.distribution_type = distribution_type
        self.distribution_args = distribution_args
        self.dataset_size = dataset_size
        self.dataset_dim = dataset_dim
        self.dataset_depth = dataset_depth
        self.seed = seed
        self.device = device

        self.data_sampler = DISTRIBUTION_FUNCTION_FACTORY[self.distribution_type](
            **self.distribution_args
        )

        if self.seed:
            # Eager dataset creation
            with fork_rng():
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                self.datasource = self.data_sampler.sample(
                    (dataset_size, dataset_depth, dataset_dim)
                )
            # copy data to device
            self.datasource = self.datasource.to(device)
        else:
            # get sampled item on indexing
            self.datasource = StochasticItems(
                self.data_sampler, (dataset_depth, dataset_dim), device
            )

    def __len__(self) -> int:
        """Gets length of dataset.

        Returns:
            int: Length of the set being sampled.
        """
        return self.dataset_size

    def __getitem__(self, index: int) -> torch.Tensor:
        """Generates a single set sample.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            torch.Tensor: Tensor containing elements with shape T x H, i.e.
                [self.dataset_depth, self.dataset_dim].
        """

        return self.datasource[index]
