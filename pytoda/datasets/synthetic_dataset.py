import random
import numpy as np
import torch
from torch.utils.data import Dataset
from paccmann_sets.utils.hyperparameters import DISTRIBUTION_FUNCTION_FACTORY


class SyntheticDataset(Dataset):
    """Generates data from a specified distribution"""

    def __init__(
        self,
        seed: int,
        distribution_type: str,
        distribution_args: dict,
        data_dim: int,
        dataset_size: int = 1,
        **kwargs
    ) -> None:
        """Constructor

        Args:
            distribution_type (str): The distribution from which data should
                be sampled. Default : normal.
            distribution_args (dict): dictionary of keyword arguments for
                the distribution specified above.
            data_dim (int): Feature size/ dimension of each sample.
            dataset_size (int): Number of samples to generate. Default 1.
        """

        super(SyntheticDataset, self).__init__()

        np.random.seed(None if seed == -1 else seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.distribution_type = distribution_type
        self.distribution_args = distribution_args
        self.dataset_size = dataset_size
        self.data_dim = data_dim

        self.data_sampler = DISTRIBUTION_FUNCTION_FACTORY[
            self.distribution_type](**self.distribution_args)

    def __len__(self) -> int:
        """Gets length of dataset.

        Returns:
            int: Length of the set being sampled.
        """
        return self.dataset_size

    def generate(self, length: int) -> torch.Tensor:
        """Generates a single set of a given length and dim.

        Args:
            length (int): Number of elements to sample of size self.data_dim.

        Returns:
            torch.Tensor: Tensor containing elements of a set with shape
                [length,self.data_dim].
        """

        return self.data_sampler.sample(
            (self.dataset_size, length, self.data_dim)
        )

    def __getitem__(self, length: int) -> torch.Tensor:
        """Generates a single set sample.

        Args:
            length (int): Number of elements to sample of size self.data_dim.

        Returns:
            torch.Tensor: Tensor containing elements of a set with shape
                [length,self.data_dim].
        """

        return self.generate(length)
