from dataclasses import dataclass
from typing import Optional, Tuple, Union

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
    shape: Union[torch.Size, Tuple[int]]
    device: torch.device

    def __getitem__(self, index: Any) -> Tensor:
        """Samples an item.

        Args:
            index (Any): is ignored.

        Returns:
            Tensor: sampled from distribution with given shape.
        """
        return self.distribution.sample(self.shape).to(self.device)


class DistributionalDataset(Dataset):
    """Generates samples from a specified distribution."""

    def __init__(
        self,
        dataset_size: int,
        item_shape: Tuple[int],
        distribution_type: str = 'normal',
        distribution_args: dict = {'loc': 0.0, 'scale': 1.0},
        seed: Optional[int] = None,
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
    ) -> None:
        """Dataset of synthetic samples from a specified distribution with given shape.

        Args:
            dataset_size (int): Number of samples to generate (N).
            item_shape (Tuple[int]): The shape of each item tensor returned on
                indexing the dataset. For example for 2D items with timeseries of
                3 timesteps and 5 features: (3, 5)
            distribution_type (str): The distribution from which data should
                be sampled. Defaults to 'normal'. For full list see:
                ``pytoda.utils.factories.DISTRIBUTION_FUNCTION_FACTORY``
            distribution_args (dict): dictionary of keyword arguments for
                the distribution specified above.
            seed (Optional[int]): If passed, all samples are generated once with
                this seed (using a local RNG only). Defaults to None, where
                individual samples are generated when the DistributionalDataset is
                indexed (using the global RNG).
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
        """

        super(DistributionalDataset, self).__init__()

        if distribution_type not in DISTRIBUTION_FUNCTION_FACTORY.keys():
            raise KeyError(
                f'distribution_type was {distribution_type}, should be from '
                f'{DISTRIBUTION_FUNCTION_FACTORY.keys()}.'
            )

        self.distribution_type = distribution_type
        self.distribution_args = distribution_args
        self.dataset_size = dataset_size
        self.item_shape = item_shape
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

                self.datasource = self.data_sampler.sample((dataset_size, *item_shape))
            # copy data to device
            self.datasource = self.datasource.to(device)
        else:
            # get sampled item on indexing
            self.datasource = StochasticItems(
                self.data_sampler, self.item_shape, device
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
