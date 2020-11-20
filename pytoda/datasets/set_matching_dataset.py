import torch
from typing import Tuple
from pytoda.datasets.base_set_matching import BaseSetMatchingDataset


class SampleSetData(BaseSetMatchingDataset):
    """Dataset class for the case where set_2 is sampled."""

    def __init__(
        self,
        dataset1: torch.Tensor,
        dataset2: torch.Tensor,
        max_set_length: int,
        min_set_length: int,
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
            dataset1 (Dataset): Object containing torch.utils.data.Dataset or child classes that represents samples of set_1.
            dataset2 (Dataset): Object containing torch.utils.data.Dataset or child classes that represents samples of set_2.
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
        super().__init__(
            max_set_length,
            min_set_length,
            vary_set_length,
            cost_metric,
            cost_metric_args,
            seed,
            device,
        )
        # Setup
        if not isinstance(seed, int):
            raise TypeError(f'Seed should be int, was {type(seed)}')
        self.seed = seed
        self.set_seed(seed)

        self.device = device

        # Setup dataset
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.max_set_length = max_set_length

        self.set_get_set_2_element()
        self.set_crop_set_lengths()

    def _set_get_set_2_element(self, idx) -> None:
        """
        Sets the function to get the elements from the second set.
        """
        # Needed in crop_set_lengths, this is a dummy in case of no permutation
        self.permutation = torch.arange(self.max_set_length)
        # Simply sample from the second dataset
        return self.dataset2[idx]

    def __len__(self) -> int:
        """Gets length of dataset.

        Returns:
            int: Length of the dataset being sampled.
        """
        return len(self.dataset1)

    def __getitem__(self, index: int) -> Tuple:
        """Generates one sample from the dataset.

        Args:
            index (int): The index to be sampled.

        Returns:
            Tuple : Tuple containing sampled set1, sampled set2, hungarian
                matching indices of set1 vs set2 and set2 vs set1.
        """

        set_1 = self.dataset1[index]
        set_2 = self._set_get_set_2_element(index)
        idx_set_1, idx_set_2 = super().set_crop_set_lengths(index)
        targets_12, targets_21 = super().get_targets(set_1, set_2)
        return set_1[idx_set_1, :], set_2[idx_set_2, :], targets_12, targets_21


class PermuteSetData(BaseSetMatchingDataset):
    """Dataset class for the case where set_2 is sampled."""

    def __init__(
        self,
        dataset1: torch.Tensor,
        noise_std: float,
        max_set_length: int,
        min_set_length: int,
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
            dataset1 (Dataset): Object containing torch.utils.data.Dataset or child classes that represents samples of set_1.
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
        super().__init__(
            max_set_length,
            min_set_length,
            vary_set_length,
            cost_metric,
            cost_metric_args,
            seed,
            device,
        )
        # Setup
        if not isinstance(seed, int):
            raise TypeError(f'Seed should be int, was {type(seed)}')
        self.seed = seed
        self.set_seed(seed)

        self.device = device

        # Setup dataset
        self.dataset1 = dataset1

        self.max_set_length = max_set_length

        self.noise = torch.distributions.normal.Normal(loc=0, scale=noise_std)

    def _set_get_set_2_element(self, set_1, idx) -> None:
        """
        Sets the function to get the elements from the second set.
        """

        def _add_noise(idx):
            super().set_seed(self.seed + idx)
            return self.noise.sample(set_1.size())

        super().set_seed(self.seed + idx)
        # Save the used permutation (needed for cropping)
        self.permutation = torch.randperm(self.max_set_length)
        return set_1[self.permutation, :] + _add_noise(set_1, idx)

    def __len__(self) -> int:
        """Gets length of dataset.

        Returns:
            int: Length of the dataset being sampled.
        """
        return len(self.dataset1)

    def __getitem__(self, index: int) -> Tuple:
        """Generates one sample from the dataset.

        Args:
            index (int): The index to be sampled.

        Returns:
            Tuple : Tuple containing sampled set1, sampled set2, hungarian
                matching indices of set1 vs set2 and set2 vs set1.
        """

        set_1 = self.dataset1[index]
        set_2 = self._set_get_set_2_element(set_1, index)
        idx_set_1, idx_set_2 = super().set_crop_set_lengths(index)
        targets_12, targets_21 = super().get_targets(set_1, set_2)
        return set_1[idx_set_1, :], set_2[idx_set_2, :], targets_12, targets_21


# class SetMatchingDataset(Dataset):
#     """Dataset class for set matching task."""

#     def __init__(
#         self,
#         *datasets: Dataset,
#         permute: bool = False,
#         vary_set_length: bool = False,
#         cost_metric: str = 'p-norm',
#         cost_metric_args: dict = {'p': 2},
#         seed: int = -1,
#         device: torch.device = (
#             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         ),
#     ):
#         """Constructor.

#         Args:
#             *datasets (Dataset): Flexible number of positional arguments, each
#                 containing torch.utils.data.Dataset or child classes.
#             permute (bool): Whether the elements of the second set are identical
#                 to the first with permuted order, or not. Defaults to False,
#                 implying that 2 datasets should be given. If True, one dataset
#                 should be given only.
#             vary_set_length (bool): Whether or not the number of elements per
#                 set should vary at runtime sampling. Defaults to False.
#             cost_metric (str): Cost metric to use when calculating the
#                 pairwise distance matrix.
#             cost_metric_args (str): Arguments for the cost metric in the
#                 right order, as specified in the function.
#             seed (int): Seed used to get the permutation in `permute` and the set
#                 length via `vary_set_length`. Hence, if both are False, the seed has
#                 no effect. Seed defaults to -1, meaning no seed is used (sampling at
#                 runtime). NOTE: This seed does not refer to stochasticity in the
#                 underlying Dataset objects.
#             device (torch.device): device where the tensors are stored.
#                 Defaults to gpu, if available.
#         """
#         # Setup
#         if not isinstance(seed, int):
#             raise TypeError(f'Seed should be int, was {type(seed)}')
#         self.seed = seed
#         self.set_seed(seed)

#         if cost_metric not in METRIC_FUNCTION_FACTORY.keys():
#             raise KeyError(
#                 f'cost_metric was {cost_metric}, should be from '
#                 f'{METRIC_FUNCTION_FACTORY.keys()}.'
#             )
#         self.cost_metric = cost_metric
#         self.cost_metric_args = cost_metric_args
#         self.get_cost_matrix = METRIC_FUNCTION_FACTORY[cost_metric](**cost_metric_args)

#         self.device = device
#         self.permute = permute
#         self.vary_set_length = vary_set_length

#         self.test_datasets(datasets)
#         # Setup dataset
#         self.datasets = datasets
#         self.max_set_length = self.datasets[0][0].shape[0]

#         self.set_get_set_2_element()
#         self.set_crop_set_lengths()

#     def test_datasets(self, datasets) -> None:
#         """Tests on dataset instances."""

#         if self.permute and len(datasets) != 1:
#             raise ValueError(
#                 f'If permute is used 1 dataset should be passed, got {len(datasets)}.'
#             )
#         if not self.permute and len(datasets) != 2:
#             raise ValueError(
#                 f'If permute is False, 2 datasets should be passed, got {len(datasets)}.'
#             )

#         # Check types
#         for d in datasets:
#             if not isinstance(d, Dataset):
#                 raise TypeError(f'Expected Dataset got {type(d)}')

#         # Check lengths
#         if len(datasets[0]) != len(datasets[-1]):
#             raise ValueError(
#                 'Length of datasets should be identical, was '
#                 f'{len(datasets[0])} and {len(datasets[-1])}.'
#             )

#         # Check shapes
#         s1 = datasets[0][0]
#         s2 = datasets[-1][0]
#         if s1.shape != s2.shape:
#             raise ValueError(
#                 f'Shapes of dataset samples must match, were {s1.shape} and {s2.shape}.'
#             )

#         if len(s1.shape) != 2:
#             raise ValueError(f'Dataset samples should be 2D, was {s1.shape}.')
#         if s1.shape[1] < 2:
#             raise ValueError('Sets should contain at least 2 elements.')

#     def set_seed(self, seed) -> None:
#         """Sets random seed if self.seed > -1.

#         Args:
#             seed (int): Sets the torch random seed. NOTE: Set is only being set if
#                 the global (class-wide) seed allows for that.
#         """
#         if self.seed > -1:
#             torch.manual_seed(seed)
#             torch.cuda.manual_seed(seed)
#             torch.cuda.manual_seed_all(seed)

#     def set_get_set_2_element(self) -> None:
#         """
#         Sets the function to get the elements from the second set.
#         """

#         if not self.permute:
#             # Simply sample from the second dataset
#             self.get_set_2_element = lambda set_1, idx: self.datasets[-1][idx]

#             # Needed in crop_set_lengths, this is a dummy in case of no permutation
#             self.permutation = torch.arange(self.max_set_length)
#         else:
#             # Set the seed (if applicable) and then perform the permutation.
#             def _get_set_2_element(set_1, idx):
#                 self.set_seed(self.seed + idx)
#                 # Save the used permutation (needed for cropping)
#                 self.permutation = torch.randperm(self.max_set_length)
#                 return set_1[self.permutation, :]

#             self.get_set_2_element = _get_set_2_element

#     def set_crop_set_lengths(self) -> None:
#         """
#         Sets the function to crop the number of elements per set.
#         """

#         if not self.vary_set_length:
#             self.crop_set_lengths = lambda set_1, set_2, idx: (set_1, set_2)
#         else:
#             # Set the seed (if applicable) and then randomly crop some elements
#             # from one set and remove the correct ones from the other.
#             def _crop_set_lengths(set_1, set_2, idx):
#                 self.set_seed(self.seed + idx)
#                 num_idxs = torch.randint(2, self.max_set_length + 1, (1,))
#                 keep_idxs_1 = torch.randperm(self.max_set_length)[:num_idxs]

#                 # This does exactly what x == y would do if x would be a Tensor and
#                 # y a int, with the extension that y is an array.
#                 keep_idxs_2 = torch.any(
#                     torch.stack(
#                         list(map(lambda x: x == self.permutation, keep_idxs_1))
#                     ),
#                     axis=0,
#                 )
#                 return set_1[keep_idxs_1, :], set_2[keep_idxs_2, :]

#             self.crop_set_lengths = _crop_set_lengths

#     def get_targets(self, set_1: torch.Tensor, set_2: torch.Tensor) -> Tuple:
#         """Compute targets for one training sample

#         Args:
#             set_1 (torch.Tensor): Tensor with elements of set_1.
#             set_2 (torch.Tensor): Tensor with elements of set_2.

#         Returns:
#             Tuple: Tuple containing hungarian matching indices of set1 vs set2 and
#                 set2 vs set1.
#         """

#         cost_matrix = self.get_cost_matrix(set_1, set_2)

#         matrix = torch.zeros_like(cost_matrix)
#         rows, cols = linear_sum_assignment(cost_matrix)
#         matrix[rows, cols] = 1
#         idx_12 = torch.from_numpy(cols)
#         idx_21 = torch.nonzero(matrix.t(), as_tuple=True)[1]

#         return idx_12, idx_21

#     def __len__(self) -> int:
#         """Gets length of dataset.

#         Returns:
#             int: Length of the dataset being sampled.
#         """
#         return len(self.datasets[0])

#     def __getitem__(self, index: int) -> Tuple:
#         """Generates one sample from the dataset.

#         Args:
#             index (int): The index to be sampled.

#         Returns:
#             Tuple : Tuple containing sampled set1, sampled set2, hungarian
#                 matching indices of set1 vs set2 and set2 vs set1.
#         """

#         set_1 = self.datasets[0][index]
#         set_2 = self.get_set_2_element(set_1, index)
#         set_1, set_2 = self.crop_set_lengths(set_1, set_2, index)
#         targets_12, targets_21 = self.get_targets(set_1, set_2)
#         return set_1, set_2, targets_12, targets_21

