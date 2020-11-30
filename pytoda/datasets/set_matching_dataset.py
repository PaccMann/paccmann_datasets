from typing import Tuple, Optional

import torch
from torch import nn
from torch.random import fork_rng
from scipy.optimize import linear_sum_assignment
from torch.functional import Tensor
from torch.utils.data import Dataset

from pytoda.datasets.utils.factories import METRIC_FUNCTION_FACTORY


def hungarian_assignment(
    set_reference: torch.Tensor,
    set_matching: torch.Tensor,
    cost_metric_function: nn.Module,
) -> Tuple:
    """Compute targets for one training sample.

    Args:
        set_reference (torch.Tensor): Tensor with elements of set_reference.
        set_matching (torch.Tensor): Tensor with elements of set_matching.
        TODO cost_metric_function

    Returns:
        Tuple: Tuple containing hungarian matching indices of set1 vs set2 and
            set2 vs set1.
    """

    cost_matrix = cost_metric_function(set_reference, set_matching)

    matrix = torch.zeros_like(cost_matrix)
    rows, cols = linear_sum_assignment(cost_matrix)
    matrix[rows, cols] = 1
    idx_12 = torch.from_numpy(cols)
    idx_21 = torch.nonzero(matrix.t(), as_tuple=True)[1]

    return idx_12, idx_21


def get_subsampling_indexes(
    min_set_length: int, max_set_length: int, permutation: Tensor, shuffle=True
) -> Tuple[Tensor, Tensor]:
    """Return indexers to remove random elements of an item and it's permutation.

    Args:
        min_set_length (int): minimal number of elements.
        max_set_length (int): maximum number of elements.
        permutation (Tensor): tensor of integers defining a permutation, that are
            indices of a range in arbitrary order.
        shuffle (bool): the first returned indexer also shuffles the elements.

    Returns:
        Tuple[Tensor, Tensor]:
            a Tensor of integers for indexing a subset of elements (shuffled or not).
            a Tensor of integers for indexing the same elements in a permuted item.
    """

    length = torch.randint(min_set_length, max_set_length + 1, (1,))
    indexes_reference = torch.randperm(max_set_length)[:length]
    if not shuffle:
        indexes_reference = indexes_reference.sort().values

    # TODO This does exactly what x == y would do if x would be a Tensor and
    # y a int, with the extension that y is an array.
    indexes_matching = torch.tensor(
        [index for index, value in enumerate(permutation) if value in indexes_reference]
    )
    return indexes_reference, indexes_matching


class BaseSetMatchingDataset(Dataset):
    """Base class for set matching datasets."""

    dataset: Dataset

    def __init__(
        self,
        max_set_length: int,
        min_set_length: int,
        cost_metric: str = 'p-norm',
        cost_metric_args: dict = {'p': 2},
        seed: Optional[int] = None,
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
    ):
        """Base Class for set matching datasets.

        Args:
            TODO min max set length docs
            cost_metric (str): Cost metric to use when calculating the
                pairwise distance matrix.
            cost_metric_args (str): Arguments for the cost metric in the
                right order, as specified in the function.
            seed (Optional[int]): TODO explain None to use no seed
                Seed used to get the permutation in `permute` and the set
                length via `vary_set_length`. Hence, if both are False, the seed has
                no effect. Seed defaults to -1, meaning no seed is used (sampling at
                runtime). NOTE: This seed does not refer to stochasticity in the
                underlying Dataset objects.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.

        Note:
            requires child classes to set the `dataset` attribute.
        """
        # test that dataset is set (by child class)
        len(self.dataset)
        # Setup
        self.seed = seed

        if cost_metric not in METRIC_FUNCTION_FACTORY.keys():
            raise KeyError(
                f'cost_metric was {cost_metric}, should be from '
                f'{METRIC_FUNCTION_FACTORY.keys()}.'
            )
        self.cost_metric = cost_metric
        self.cost_metric_args = cost_metric_args
        self.get_cost_matrix = METRIC_FUNCTION_FACTORY[cost_metric](**cost_metric_args)

        self.device = device

        self.max_set_length = max_set_length
        self.min_set_length = min_set_length

    @property
    def permutation(self) -> Tensor:
        raise NotImplementedError

    def get_matching_set(self, index: int, reference_set: Tensor) -> Tensor:
        raise NotImplementedError

    def __len__(self) -> int:
        """Gets length of dataset.

        Returns:
            int: Length of the dataset being sampled.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple:
        """Generates one sample from the dataset.

        Args:
            index (int): The index to be sampled.

        Returns:
            Tuple : Tuple containing sampled set1, sampled set2, hungarian
                matching indices of set1 vs set2 and set2 vs set1.
        """

        set_reference = self.dataset[index]

        enable_fork_rng = self.seed is not None
        with fork_rng(enable=enable_fork_rng):
            if enable_fork_rng:
                seed = self.seed + index
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            indexes_reference, indexes_matching = get_subsampling_indexes(
                self.min_set_length,
                self.max_set_length,
                self.permutation,
                shuffle=True,
            )

            set_matching = self.get_matching_set(index, set_reference)

        cropped_set_reference = set_reference[indexes_reference, :]
        cropped_set_matching = set_matching[indexes_matching, :]
        targets_12, targets_21 = hungarian_assignment(
            cropped_set_reference,
            cropped_set_matching,
            cost_metric_function=self.get_cost_matrix,
        )
        return (
            cropped_set_reference,
            cropped_set_matching,
            targets_12,
            targets_21,
        )


class SampleSetData(BaseSetMatchingDataset):
    """Dataset class for the case where set_matching is sampled."""

    def __init__(
        self,
        dataset: Dataset,
        dataset_to_match: Dataset,
        max_set_length: int,
        min_set_length: int,
        cost_metric: str = 'p-norm',
        cost_metric_args: dict = {'p': 2},
        seed: int = -1,  # TODO
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
    ):

        """Constructor. TODO

        Args:
            dataset (Dataset): Object containing torch.utils.data.Dataset or child classes that represents samples of reference_set.
            dataset_to_match (Dataset): Object containing torch.utils.data.Dataset or child classes that represents samples of set_matching.
            cost_metric (str): Cost metric to use when calculating the
                pairwise distance matrix.
            cost_metric_args (str): Arguments for the cost metric in the
                right order, as specified in the function.
            seed (int): Seed used to get the permutation in `permute` and the set
                length via `vary_set_length`. Hence, if both are False, the seed has
                no effect. Seed defaults to -1, meaning no seed is used (sampling at
                runtime). NOTE: This seed does not refer to stochasticity in the
                underlying Dataset objects.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
        """
        # Setup dataset
        self.dataset = dataset
        self.dataset_to_match = dataset_to_match
        self.dummy_permutation = torch.arange(self.max_set_length)
        super().__init__(
            max_set_length,
            min_set_length,
            cost_metric,  # TODO
            cost_metric_args,
            seed,
            device,
        )

    @property
    def permutation(self) -> Tensor:
        return self.dummy_permutation

    def get_matching_set(self, index: int, reference_set: Tensor) -> Tensor:
        return self.dataset_to_match[index]


class PermuteSetData(BaseSetMatchingDataset):
    """Dataset class for the case where set_matching is sampled."""

    def __init__(
        self,
        dataset: Dataset,
        noise_std: float,
        max_set_length: int,
        min_set_length: int,
        cost_metric: str = 'p-norm',
        cost_metric_args: dict = {'p': 2},
        seed: int = -1,  # TODO
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
    ):

        """Constructor. TODO

        Args:
            dataset (Dataset): Object containing torch.utils.data.Dataset or child classes that represents samples of set_reference.
            cost_metric (str): Cost metric to use when calculating the
                pairwise distance matrix.
            cost_metric_args (str): Arguments for the cost metric in the
                right order, as specified in the function.
            seed (int): TODO
                Seed used to get the permutation in `permute` and the set
                length via `vary_set_length`. Hence, if both are False, the seed has
                no effect. Seed defaults to -1, meaning no seed is used (sampling at
                runtime). NOTE: This seed does not refer to stochasticity in the
                underlying Dataset objects.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
        """
        # Setup dataset
        self.dataset = dataset
        self.noise = torch.distributions.normal.Normal(loc=0, scale=noise_std)
        super().__init__(
            # needs to be number of elements in an item, depends on dataset!
            max_set_length,
            # is user choice
            min_set_length,
            cost_metric,
            cost_metric_args,
            seed,
            device,
        )

    @property
    def permutation(self) -> Tensor:
        return torch.randperm(self.max_set_length)

    def get_matching_set(self, index: int, reference_set: Tensor) -> Tensor:
        additive_noise = self.noise.sample(reference_set.size())
        return reference_set[self.permutation, :] + additive_noise
