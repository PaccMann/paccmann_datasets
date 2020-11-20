from typing import Tuple
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset

from pytoda.datasets.utils.factories import METRIC_FUNCTION_FACTORY


class BaseSetMatchingDataset(Dataset):
    """Base class for set matching datasets."""

    def __init__(
        self,
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
        self.vary_set_length = vary_set_length

        self.max_set_length = max_set_length
        self.min_set_length = min_set_length

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

    def set_crop_set_lengths(self, idx) -> None:
        """
        Sets the function to crop the number of elements per set.
        """
        self.set_seed(self.seed + idx)
        num_idxs = torch.randint(self.min_set_length, self.max_set_length + 1, (1,))
        keep_idxs_1 = torch.randperm(self.max_set_length)[:num_idxs]

        # This does exactly what x == y would do if x would be a Tensor and
        # y a int, with the extension that y is an array.
        keep_idxs_2 = torch.any(
            torch.stack(list(map(lambda x: x == self.permutation, keep_idxs_1))),
            axis=0,
        )
        return keep_idxs_1, keep_idxs_2

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
