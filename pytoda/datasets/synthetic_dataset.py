import torch
from torch.utils.data import Dataset

from pytoda.datasets.utils.factories import DISTRIBUTION_FUNCTION_FACTORY


class SyntheticDataset(Dataset):
    """Generates 2D samples from a specified distribution."""

    def __init__(
        self,
        dataset_size: int,
        dataset_dim: int,
        dataset_depth: int = 1,
        distribution_type: str = 'normal',
        distribution_args: dict = {'loc': 0.0, 'scale': 1.0},
        seed: int = -1,
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
    ) -> None:
        """Constructor.

        Args:
            dataset_size (int): Number of samples to generate (N).
            dataset_dim (int): Feature size / dimension of each sample (H).
            dataset_depth (int): Length of time series per sample (T). This is to
                support 2D samples. Sampling from __getitem__ will have shape
                T x H. T defaults to 1.
            distribution_type (str): The distribution from which data should
                be sampled. Default : normal. For full list see:
                pytoda.utils.factories.DISTRIBUTION_FUNCTION_FACTORY
            distribution_args (dict): dictionary of keyword arguments for
                the distribution specified above.
            seed (int): Seed used for the dataset generator. Defaults to -1, meaning
                no seed is used (sampling at runtime).
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
        """

        super(SyntheticDataset, self).__init__()

        if not isinstance(seed, int):
            raise TypeError(f'Seed should be int, was {type(seed)}')

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
        self.device = device

        self.data_sampler = DISTRIBUTION_FUNCTION_FACTORY[self.distribution_type](
            **self.distribution_args
        )

        self.seed = seed
        if seed > -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # Eager dataset creation
            self.data = self.data_sampler.sample(
                (dataset_size, dataset_depth, dataset_dim)
            )
            # self.transform = lambda x: x

        else:
            self.data = torch.zeros(dataset_size, dataset_depth, dataset_dim)
            # self.transform = lambda x: x + self.data_sampler.sample(x.shape).to(device)

        # Copy data to device
        self.data = self.data.to(device)

    def _transform(self, x):
        if self.seed > -1:
            return x
        else:
            sample = self.data_sampler.sample(x.shape).to(self.device)
            return x + sample

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

        return self._transform(self.data[index])
