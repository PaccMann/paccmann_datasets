from typing import Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset

from pytoda.datasets.utils.factories import METRIC_FUNCTION_FACTORY


class SetMatchingDataset(Dataset):
    """Dataset class for set matching task."""

    def __init__(
        self,
        *datasets: Dataset,
        permute: bool = False,
        vary_set_length: bool = False,
        cost_metric: str = 'p-norm',
        cost_metric_args: dict = {'p': 2},
        seed: int = -1,
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
    ):
        """Constructor.

        Args:
            *datasets (Dataset): Flexible number of positional arguments, each
                containing torch.utils.data.Dataset or child classes.
            permute (bool): Whether the elements of the second set are identical
                to the first with permuted order, or not. Defaults to False,
                implying that 2 datasets should be given. If True, one dataset
                should be given only.
            vary_set_length (bool): Whether or not the number of elements per
                set should vary at runtime sampling. Defaults to False.
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
        # Setup
        if not isinstance(seed, int):
            raise TypeError(f'Seed should be int, was {type(seed)}')
        self.seed = seed
        self.set_seed(seed)

        if cost_metric not in METRIC_FUNCTION_FACTORY.keys():
            raise KeyError(
                f'cost_metric was {cost_metric}, should be from '
                f'{METRIC_FUNCTION_FACTORY.keys()}.'
            )
        self.cost_metric = cost_metric
        self.cost_metric_args = cost_metric_args
        self.get_cost_matrix = METRIC_FUNCTION_FACTORY[cost_metric](**cost_metric_args)

        self.device = device
        self.permute = permute
        self.vary_set_length = vary_set_length

        self.test_datasets(datasets)
        # Setup dataset
        self.datasets = datasets
        self.max_set_length = self.datasets[0][0].shape[0]

        self.set_get_set_2_element()
        self.set_crop_set_lengths()

    def test_datasets(self, datasets) -> None:
        """Tests on dataset instances."""

        if self.permute and len(datasets) != 1:
            raise ValueError(
                f'If permute is used 1 dataset should be passed, got {len(datasets)}.'
            )
        if not self.permute and len(datasets) != 2:
            raise ValueError(
                f'If permute is False, 2 datasets should be passed, got {len(datasets)}.'
            )

        # Check types
        for d in datasets:
            if not isinstance(d, Dataset):
                raise TypeError(f'Expected Dataset got {type(d)}')

        # Check lengths
        if len(datasets[0]) != len(datasets[-1]):
            raise ValueError(
                'Length of datasets should be identical, was '
                f'{len(datasets[0])} and {len(datasets[-1])}.'
            )

        # Check shapes
        s1 = datasets[0][0]
        s2 = datasets[-1][0]
        if s1.shape != s2.shape:
            raise ValueError(
                f'Shapes of dataset samples must match, were {s1.shape} and {s2.shape}.'
            )

        if len(s1.shape) != 2:
            raise ValueError(f'Dataset samples should be 2D, was {s1.shape}.')
        if s1.shape[1] < 2:
            raise ValueError('Sets should contain at least 2 elements.')

    def set_seed(self, seed) -> None:
        """Sets random seed if self.seed > -1.

        Args:
            seed (int): Sets the torch random seed. NOTE: Set is only being set if
                the global (class-wide) seed allows for that.
        """
        if self.seed > -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def set_get_set_2_element(self) -> None:
        """
        Sets the function to get the elements from the second set.
        """

        if not self.permute:
            # Simply sample from the second dataset
            self.get_set_2_element = lambda set_1, idx: self.datasets[-1][idx]

            # Needed in crop_set_lengths, this is a dummy in case of no permutation
            self.permutation = torch.arange(self.max_set_length)
        else:
            # Set the seed (if applicable) and then perform the permutation.
            def _get_set_2_element(set_1, idx):
                self.set_seed(self.seed + idx)
                # Save the used permutation (needed for cropping)
                self.permutation = torch.randperm(self.max_set_length)
                return set_1[self.permutation, :]

            self.get_set_2_element = _get_set_2_element

    def set_crop_set_lengths(self) -> None:
        """
        Sets the function to crop the number of elements per set.
        """

        if not self.vary_set_length:
            self.crop_set_lengths = lambda set_1, set_2, idx: (set_1, set_2)
        else:
            # Set the seed (if applicable) and then randomly crop some elements
            # from one set and remove the correct ones from the other.
            def _crop_set_lengths(set_1, set_2, idx):
                self.set_seed(self.seed + idx)
                num_idxs = torch.randint(2, self.max_set_length + 1, (1,))
                keep_idxs_1 = torch.randperm(self.max_set_length)[:num_idxs]

                # This does exactly what x == y would do if x would be a Tensor and
                # y a int, with the extension that y is an array.
                keep_idxs_2 = torch.any(
                    torch.stack(
                        list(map(lambda x: x == self.permutation, keep_idxs_1))
                    ),
                    axis=0,
                )
                return set_1[keep_idxs_1, :], set_2[keep_idxs_2, :]

            self.crop_set_lengths = _crop_set_lengths

    def get_targets(self, set_1: torch.Tensor, set_2: torch.Tensor) -> Tuple:
        """Compute targets for one training sample

        Args:
            set_1 (torch.Tensor): Tensor with elements of set_1.
            set_2 (torch.Tensor): Tensor with elements of set_2.

        Returns:
            Tuple: Tuple containing hungarian matching indices of set1 vs set2 and
                set2 vs set1.
        """

        cost_matrix = self.get_cost_matrix(set_1, set_2)

        matrix = torch.zeros_like(cost_matrix)
        rows, cols = linear_sum_assignment(cost_matrix)
        matrix[rows, cols] = 1
        idx_12 = torch.from_numpy(cols)
        idx_21 = torch.nonzero(matrix.t(), as_tuple=True)[1]

        return idx_12, idx_21

    def __len__(self) -> int:
        """Gets length of dataset.

        Returns:
            int: Length of the dataset being sampled.
        """
        return len(self.datasets[0])

    def __getitem__(self, index: int) -> Tuple:
        """Generates one sample from the dataset.

        Args:
            index (int): The index to be sampled.

        Returns:
            Tuple : Tuple containing sampled set1, sampled set2, hungarian
                matching indices of set1 vs set2 and set2 vs set1.
        """

        set_1 = self.datasets[0][index]
        set_2 = self.get_set_2_element(set_1, index)
        set_1, set_2 = self.crop_set_lengths(set_1, set_2, index)
        targets_12, targets_21 = self.get_targets(set_1, set_2)
        return set_1, set_2, targets_12, targets_21


class CollatorSetMatching:
    """Contains function to pad data returned by dataloader."""

    def __init__(
        self,
        dim: int,
        max_length: int,
        batch_first: bool = True,
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
    ):
        """Constructor.

        Args:
            dim (int): Dimension of the data.
            max_length (int): Maximum set length.
            batch_first (bool, optional): Whether batch size is the first
                dimension or not. Defaults to True.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
        """
        super(CollatorSetMatching, self).__init__()
        self.dim = dim
        self.max_len = max_length
        self.batch_first = eval(batch_first)
        self.device = device

    def __call__(self, DataLoaderBatch: Tuple) -> Tuple:
        """Collate function for batch-wise padding of samples.

        Args:
            DataLoaderBatch (Tuple): Tuple of tensors returned by get_item of the
            dataset class.

        Returns:
            Tuple: Tuple of padded input tensors and tensor of set lengths.
        """

        batch_size = len(DataLoaderBatch)
        batch_split = list(zip(*DataLoaderBatch))

        sets1, sets2, targs12, targs21 = (
            batch_split[0],
            batch_split[1],
            batch_split[2],
            batch_split[3],
        )

        lengths = list(map(len, sets1))
        pad_token = 10.0

        padded_sets1 = torch.full(
            (batch_size, self.max_len, self.dim), pad_token, device=self.device
        )
        padded_sets2 = torch.full(
            (batch_size, self.max_len, self.dim), pad_token, device=self.device
        )
        targets12 = np.tile(np.arange(self.max_len), (batch_size, 1))
        targets21 = np.tile(np.arange(self.max_len), (batch_size, 1))
        targets12 = torch.from_numpy(targets12).to(self.device)
        targets21 = torch.from_numpy(targets21).to(self.device)

        for i, l in enumerate(lengths):
            padded_sets1[i, 0:l, :] = sets1[i][0:l, :]
            padded_sets2[i, 0:l, :] = sets2[i][0:l, :]

            targets12[i, 0:l] = targs12[i][:]
            targets21[i, 0:l] = targs21[i][:]

        if self.batch_first is False:
            padded_sets1, padded_sets2 = (
                padded_sets1.permute(1, 0, 2),
                padded_sets2.permute(1, 0, 2),
            )

        return padded_sets1, padded_sets2, targets12, targets21, torch.tensor(lengths)
