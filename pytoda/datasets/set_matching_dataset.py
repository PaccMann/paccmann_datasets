from typing import Tuple, Optional

import torch
from torch import nn
from torch.random import fork_rng
from scipy.optimize import linear_sum_assignment
from torch.functional import Tensor
from torch.utils.data import Dataset


def hungarian_assignment(
    set_reference: torch.Tensor,
    set_matching: torch.Tensor,
    cost_metric_function: nn.Module,
) -> Tuple:
    """Compute targets for one training sample.

    Args:
        set_reference (torch.Tensor): Tensor with elements of set_reference.
        set_matching (torch.Tensor): Tensor with elements of set_matching.
        cost_metric_function (nn.Module): Function wrapped as an nn.Module that
            computes the metric used in constructing the cost matrix.

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
        min_set_length (int): Minimum number of elements in the set.
        max_set_length (int): Maximum number of elements in the set.
        permutation (Tensor): Tensor of integers defining a permutation, that are
            indices of a range in arbitrary order.
        shuffle (bool): The first returned indexer also shuffles the elements.

    Returns:
        Tuple[Tensor, Tensor]:
            A Tensor of integers for indexing a subset of elements (shuffled or not).
            A Tensor of integers for indexing the same elements in a permuted item.
    """

    length = torch.randint(min_set_length, max_set_length + 1, (1,))
    indexes_reference = torch.randperm(max_set_length)[:length]
    if not shuffle:
        indexes_reference = indexes_reference.sort().values

    # This identifies indexes in set_matching that correspond to indexes_reference.
    # By doing so, we ensure that in the 'permute' case, the correct elements are retained during random cropping.
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
        cost_metric_function: nn.Module,
        seed: Optional[int] = None,
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
    ):
        """Base Class for set matching datasets.

        Args:
            max_set_length (int): Maximum number of elements in the set. This
                should be equal to the item length of the distributional dataset
                set by the child class.
            min_set_length (int): Minimum number of elements required in the set.
                This should be equal to max_set_length if varying set lengths are not desired.
            cost_metric_function (nn.Module): Function wrapped as an nn.Module that
                computes the metric used in constructing the cost matrix.
            seed (Optional[int]): If passed, all samples are generated once with
                this seed (using a local RNG only). Defaults to None, where
                individual samples are generated when the DistributionalDataset is
                indexed (using the global RNG).
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.

        Note:
            requires child classes to set the `dataset` attribute.
        """
        # test that dataset is set (by child class)
        len(self.dataset)
        # Setup
        self.seed = seed

        self.get_cost_matrix = cost_metric_function

        self.device = device

        self.max_set_length = max_set_length
        self.min_set_length = min_set_length

    @property
    def permutation(self) -> Tensor:
        """Class attribute that defines the permutation to use in creating set_matching.

        Raises:
            NotImplementedError: Not implemented by the base class. Attribute overwritten
                by the child class.
        """
        raise NotImplementedError

    def get_matching_set(self, index: int, reference_set: Tensor) -> Tensor:
        """Gets the corresponding set to match to the reference set.

        Args:
            index (int): The index to be sampled.
            reference_set (Tensor): Tensor that represents samples of the reference set.

        Raises:
            NotImplementedError: Not implemented by the base class. Function overwritten
                by the child class.
        """
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
        cost_metric_function: nn.Module,
        seed: Optional[int] = None,
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
    ):

        """Constructor that initialises BaseSetMatchingDataset class and the inheriting SampleSetData class.

        Args:
            dataset (Dataset): Object containing torch.utils.data.Dataset or child classes that represents samples of set_reference.
            dataset_to_match (Dataset): Object containing torch.utils.data.Dataset or child classes that represents samples of set_matching.
            max_set_length (int): Maximum number of elements in the set. This
                should be equal to the item length of the distributional dataset
                passed into this class.
            min_set_length (int): Minimum number of elements required in the set.
                Set it equal to max_set_length if varying set lengths are not desired.
            cost_metric_function (nn.Module): Function wrapped as an nn.Module that
                computes the metric used in constructing the cost matrix.
            seed (Optional[int]): If passed, all samples are generated once with
                this seed (using a local RNG only). Defaults to None, where
                individual samples are generated when the DistributionalDataset is
                indexed (using the global RNG).
            device (torch.device): Device where the tensors are stored.
                Defaults to gpu, if available.
        """
        # Setup dataset
        self.dataset = dataset
        self.dataset_to_match = dataset_to_match
        self.dummy_permutation = torch.arange(self.max_set_length)
        super().__init__(
            max_set_length, min_set_length, cost_metric_function, seed, device,
        )

    @property
    def permutation(self) -> Tensor:
        """Class attribute that defines the permutation to use in creating set_matching.

        Returns:
            Tensor: A fixed tensor containing the range of max_set_length.
        """
        return self.dummy_permutation

    def get_matching_set(self, index: int, reference_set: Tensor) -> Tensor:
        """Gets the corresponding set to match to the reference set.

        Args:
            index (int): The index to be sampled.
            reference_set (Tensor): Tensor that represents samples of the reference set.

        Returns:
            Tensor: Tensor of the corresponding matching set.
        """
        return self.dataset_to_match[index]


class PermuteSetData(BaseSetMatchingDataset):
    """Dataset class for the case where set_matching is sampled."""

    def __init__(
        self,
        dataset: Dataset,
        noise_std: float,
        max_set_length: int,
        min_set_length: int,
        cost_metric_function: nn.Module,
        seed: Optional[int] = None,
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
    ):

        """Constructor that initialises BaseSetMatchingDataset class and the inheriting PermuteSetData class.

        Args:
            dataset (Dataset): Object containing torch.utils.data.Dataset or child classes that represents samples of set_reference.
            noise_std (float): Standard deviation to use in generating noise from a normal distribution with mean 0.
            max_set_length (int): Maximum number of elements in the set. This
                should be equal to the item length of the distributional dataset
                passed into this class.
            min_set_length (int): Minimum number of elements required in the set.
                Set it equal to max_set_length if varying set lengths are not desired.
            cost_metric_function (nn.Module): Function wrapped as an nn.Module that
                computes the metric used in constructing the cost matrix.
            seed (Optional[int]): If passed, all samples are generated once with
                this seed (using a local RNG only). Defaults to None, where
                individual samples are generated when the DistributionalDataset is
                indexed (using the global RNG).
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
            cost_metric_function,
            seed,
            device,
        )

    @property
    def permutation(self) -> Tensor:
        """Class attribute that defines the permutation to use in creating set_matching.

        Returns:
            Tensor: Tensor of a randomly generated permutation of length max_set_length.
        """
        return torch.randperm(self.max_set_length)

    def get_matching_set(self, index: int, reference_set: Tensor) -> Tensor:
        """Gets the corresponding set to match to the reference set.

        Args:
            index (int): The index to be sampled.
            reference_set (Tensor): Tensor that represents samples of the reference set.

        Returns:
            Tensor: Tensor of the permuted reference set with additive noise.
        """
        additive_noise = self.noise.sample(reference_set.size())
        return reference_set[self.permutation, :] + additive_noise
