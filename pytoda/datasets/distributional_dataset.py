from typing import Optional, Tuple, Union

import torch
from torch.random import fork_rng
from torch.utils.data import Dataset

from pytoda.types import Any, Tensor


class StochasticItems:
    """Sample an item from a distribution on the fly on indexing.

    Args:
        distribution (torch.distributions.distribution.Distribution): An instance
            of the torch distribution class to sample from. For example, for
            loc = torch.tensor(0.0,device=device), and scale=torch.tensor(1.0,device=device),
            torch.distributions.normal.Normal(loc,scale), so that
            calling .sample() would return an item from this distribution on the specified device.
            NOTE: The arguments to the distribution should be tensors on the desired device.
            This ensure that samples are generated on this device and helps in avoiding
            an overhead in sending each sampled item to device.
        shape (torch.Size): The desired shape of each item.
    """

    def __init__(
        self,
        distribution: torch.distributions.distribution.Distribution,
        shape: Union[torch.Size, Tuple[int]],
    ):

        self.distribution = distribution
        self.shape = shape

    def __getitem__(self, index: Any) -> Tensor:
        """Samples an item.

        Args:
            index (Any): is ignored.

        Returns:
            Tensor: sampled from distribution with given shape.
        """

        return self.distribution.sample(self.shape)


class DistributionalDataset(Dataset):
    """Generates samples from a specified distribution."""

    def __init__(
        self,
        dataset_size: int,
        item_shape: Tuple[int],
        distribution_function: torch.distributions.distribution.Distribution,
        seed: Optional[int] = None,
    ) -> None:
        """Dataset of synthetic samples from a specified distribution with given shape.

        Args:
            dataset_size (int): Number of items to generate (N).
            item_shape (Tuple[int]): The shape of each item tensor returned on
                indexing the dataset. For example for 2D items with timeseries of
                3 timesteps and 5 features: (3, 5)
            distribution_function (torch.distributions.distribution.Distribution):
                An instance of the distribution class from which individual data
                items can be sampled by calling the .sample() method.
                This can either be an object that is directly initialised using a method
                from torch.distributions, such as, torch.distributions.normal.Normal(loc=0.0,scale=1.0),
                or from a factory using a keyword, for example,
                DISTRIBUTION_FUNCTION_FACTORY['normal](loc=0.0, scale=1.0)
                is a valid argument since the factory (found in utils.factories.py)
                initialises the distribution class object based on a string keyword
                and passes the relevant arguments to that object.
            seed (Optional[int]): If passed, all items are generated once with
                this seed (using a local RNG only). Defaults to None, where
                individual items are generated when the DistributionalDataset is
                indexed (using the global RNG).
        """

        super(DistributionalDataset, self).__init__()

        self.dataset_size = dataset_size
        self.item_shape = item_shape
        self.seed = seed

        self.data_sampler = distribution_function

        if self.seed:
            # Eager dataset creation
            with fork_rng():
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                self.datasource = self.data_sampler.sample((dataset_size, *item_shape))

        else:
            # get sampled item on indexing
            self.datasource = StochasticItems(
                self.data_sampler,
                self.item_shape,
                torch.device('cpu'),
            )

        # copy data to device

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
