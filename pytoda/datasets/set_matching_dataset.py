import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.random import fork_rng
from torch.utils.data import Dataset

from pytoda.datasets.utils.utils import pad_item
from pytoda.types import Optional, Tensor, Tuple


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
        Tuple[Tensor, Tensor, int]:
            A Tensor of integers for indexing a subset of elements (shuffled or not).
            A Tensor of integers for indexing the same elements in a permuted item.
            Number of elements in the item (length).
    """
    length = torch.randint(min_set_length, max_set_length + 1, (1,))
    if min_set_length != max_set_length:
        indexes_reference = torch.randperm(max_set_length)[:length]
        if not shuffle:
            indexes_reference = indexes_reference.sort().values
    else:
        indexes_reference = torch.arange(max_set_length)

    # This identifies indexes in set_matching that correspond to indexes_reference.
    # By doing so, we ensure that in the 'permute' case, the correct elements are retained during random cropping.

    indexes_matching = torch.tensor(
        [index for index, value in enumerate(permutation) if value in indexes_reference]
    )

    return indexes_reference, indexes_matching, length


class SetMatchingDataset(Dataset):
    """Base class for set matching datasets."""

    dataset: Dataset

    def __init__(
        self,
        min_set_length: int,
        cost_metric_function: nn.Module,
        set_padding_value: float = 0.0,
        seed: Optional[int] = None,
        shuffle: Optional[bool] = True,
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
    ):
        """Base Class for set matching datasets, that allows returning subsets of
            varying length controlled by the passed min set length.

        Args:
            min_set_length (int): Lower bound on number of elements in the returned sets.
                This should be equal to the maximum item length of the passed dataset
                if varying set lengths are not desired.
            cost_metric_function (nn.Module): Function wrapped as an nn.Module that
                computes the metric used in constructing the cost matrix.
            set_padding_value (float): The constant value with which to pad each set.
                Defaults to 0.0. NOTE: No padding is done if min_set_length = max_set_length.
            seed (Optional[int]): If passed, all samples are generated once with
                this seed (using a local RNG only). Defaults to None, where
                individual samples are generated when the DistributionalDataset is
                indexed (using the global RNG).
            shuffle (Optional[bool]): Whether the sets should be shuffled again before subsampling.
                Adds another layer of randomness. Defaults to True.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.

        Note:
            Requires child classes to set the `dataset` attribute.
            Batches generated by DataLoader will always have the batch size in
                the first dimension.
        """
        # test that dataset is set (by child class)
        len(self.dataset)
        self.max_set_length = len(self.dataset[0])
        # Setup
        self.seed = seed

        self.get_cost_matrix = cost_metric_function

        self.device = device

        self.min_set_length = min_set_length

        self.pad = not (min_set_length == self.max_set_length)

        self.set_padding_value = set_padding_value

        self.shuffle = shuffle

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
        with fork_rng(enabled=enable_fork_rng):
            if enable_fork_rng:
                seed = self.seed + index
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            self.permute_indices = self.permutation
            indexes_reference, indexes_matching, length = get_subsampling_indexes(
                self.min_set_length,
                self.max_set_length,
                self.permute_indices,
                shuffle=self.shuffle,
            )

        set_matching = self.get_matching_set(index, set_reference)

        cropped_set_reference = set_reference[indexes_reference, :]
        cropped_set_matching = set_matching[indexes_matching, :]

        targets_12, targets_21 = hungarian_assignment(
            cropped_set_reference,
            cropped_set_matching,
            cost_metric_function=self.get_cost_matrix,
        )

        if not self.pad:
            return (
                cropped_set_reference,
                cropped_set_matching,
                targets_12,
                targets_21,
                length,
            )
        else:
            return (
                *pad_item(
                    item=(
                        cropped_set_reference,
                        cropped_set_matching,
                        targets_12,
                        targets_21,
                    ),
                    padding_modes=['constant', 'constant', 'range', 'range'],
                    padding_values=[
                        self.set_padding_value,
                        self.set_padding_value,
                        range(self.max_set_length),
                        range(self.max_set_length),
                    ],
                    max_length=self.max_set_length,
                    device=self.device,
                ),
                length,
            )


class PairedSetMatchingDataset(SetMatchingDataset):
    """Dataset class for the case where set to match is another random set."""

    def __init__(
        self,
        dataset: Dataset,
        dataset_to_match: Dataset,
        min_set_length: int,
        cost_metric_function: nn.Module,
        set_padding_value: float,
        noise_std: float = 0.0,
        seed: Optional[int] = None,
        shuffle: Optional[bool] = True,
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
    ):
        """Pairs of sets and their hungarian assignments, where the set to match
        is provided by a second dataset.

        Args:
            dataset (Dataset): Object containing torch.utils.data.Dataset or
                child classes that represents samples of set_reference.
            dataset_to_match (Dataset): Object containing torch.utils.data.Dataset
                or child classes that represents samples of set_matching.
            min_set_length (int): Lower bound on number of elements in the returned sets.
                This should be equal to the maximum item length of the passed dataset
                if varying set lengths are not desired.
            set_padding_value (float): The constant value with which to pad each set.
                Defaults to 0.0. NOTE: No padding if min_set_length = max_set_length.
            cost_metric_function (nn.Module): Function wrapped as an nn.Module that
                computes the metric used in constructing the cost matrix.
            noise_std (float, optional): Standard deviation to use in generating
                noise from a normal distribution with mean 0. Deafults to 0.0.
                Dummy variable for this class for consistency purposes.
            seed (Optional[int]): If passed, all samples are generated once with
                this seed (using a local RNG only). Defaults to None, where
                individual samples are generated when the DistributionalDataset is
                indexed (using the global RNG).
            shuffle (Optional[bool]): Whether the sets should be shuffled again before subsampling.
                Adds another layer of randomness. Defaults to True.
            device (torch.device): Device where the tensors are stored.
                Defaults to gpu, if available.
        """
        # Setup dataset
        self.dataset = dataset
        self.dataset_to_match = dataset_to_match

        super().__init__(
            min_set_length,
            cost_metric_function,
            set_padding_value,
            seed,
            shuffle,
            device,
        )
        self.dummy_permutation = torch.arange(self.max_set_length)

    @property
    def permutation(self) -> Tensor:
        """Class attribute that defines the permutation to use in creating set_matching.

        Returns:
            Tensor: A fixed tensor containing the range of max_set_length.
        """
        return self.dummy_permutation

    def get_matching_set(self, index: int, set_reference: Tensor) -> Tensor:
        """Gets the corresponding set to match to the reference set.

        Args:
            index (int): The index to be sampled.
            reference_set (Tensor): Tensor that represents samples of the reference set.

        Returns:
            Tensor: Tensor of the corresponding matching set.
        """
        return self.dataset_to_match[index]


class PermutedSetMatchingDataset(SetMatchingDataset):
    """Dataset class for the case where set to match is permuted."""

    def __init__(
        self,
        dataset: Dataset,
        min_set_length: int,
        cost_metric_function: nn.Module,
        set_padding_value: float,
        noise_std: float = 0.0,
        seed: Optional[int] = None,
        shuffle: Optional[bool] = True,
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
    ):
        """Pairs of sets and their hungarian assignments, where the set to match
        is a permutation of the given sets.

        Args:
            dataset (Dataset): Object containing torch.utils.data.Dataset or child
                classes that represents samples of set_reference.
            min_set_length (int): Lower bound on number of elements in the returned sets.
                This should be equal to the maximum item length of the passed dataset
                if varying set lengths are not desired.
            set_padding_value (float): The constant value with which to pad each set.
                Defaults to 0.0. NOTE: No padding is done if min_set_length = max_set_length.
            cost_metric_function (nn.Module): Function wrapped as an nn.Module that
                computes the metric used in constructing the cost matrix.
            noise_std (float, optional): Standard deviation to use in generating noise from
                a normal distribution with mean 0. Deafults to 0.0.
            seed (Optional[int]): If passed, all samples are generated once with
                this seed (using a local RNG only). Defaults to None, where
                individual samples are generated when the DistributionalDataset is
                indexed (using the global RNG).
            shuffle (Optional[bool]): Whether the sets should be shuffled again before subsampling.
                Adds another layer of randomness. Defaults to True.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
        """
        # Setup dataset
        self.dataset = dataset
        self.noise = torch.distributions.normal.Normal(loc=0, scale=noise_std)
        super().__init__(
            # is user choice
            min_set_length,
            cost_metric_function,
            set_padding_value,
            seed,
            shuffle,
            device,
        )

    @property
    def permutation(self) -> Tensor:
        """Class attribute that defines the permutation to use in creating set_matching.

        Returns:
            Tensor: Tensor of a randomly generated permutation of length max_set_length.
        """
        return torch.randperm(self.max_set_length)

    def get_matching_set(self, index: int, set_reference: Tensor) -> Tensor:
        """Gets the corresponding set to match to the reference set.

        Args:
            index (int): The index to be sampled.
            reference_set (Tensor): Tensor that represents samples of the reference set.

        Returns:
            Tensor: Tensor of the permuted reference set with additive noise.
        """
        enable_fork_rng = self.seed is not None
        with fork_rng(enabled=enable_fork_rng):
            if enable_fork_rng:
                seed = self.seed + index
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            additive_noise = self.noise.sample(set_reference.size())

        return set_reference[self.permute_indices, :] + additive_noise