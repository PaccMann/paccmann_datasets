import numpy as np
import random
import torch
from typing import Tuple
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment
from .synthetic_dataset import SyntheticDataset
from ..factories import METRIC_FUNCTION_FACTORY


class SetMatchingDataset(Dataset):

    def __init__(self, data_params: dict):
        """Constructor.

        Args:
            data_params (dict): Dictionary of parameters necessary for dataset
                generation. Example keys:
                    seed (int): Seed for reproducibility. Default -1 for random.
                    distribution_type (str): The distribution from which data
                        should be sampled. Default : normal.
                    distribution_args (dict): dictionary of keyword arguments
                        for the distribution specified above.
                    dataset_size (int): Number of samples to generate.
                    max_length (int): Maximum length of a set.
                    min_length (int): Minimum length of the set.
                    data_dim (int): Feature size/ dimension of each sample.
                    cost_metric (str): Cost metric to use when calculating the
                        pairwise distance matrix.
                    cost_metric_args (str): Arguments for the cost metric in the
                        right order, as specified in the function.
        """

        self.seed = data_params.get("seed", -1)

        np.random.seed(None if self.seed == -1 else self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.dataset_size = data_params.get("dataset_size", 10000)
        # total_size = sum(
        #     [
        #         data_params.get("train_size", 10000),
        #         data_params.get("valid_size", 5000),
        #         data_params.get("test_size", 10000)
        #     ]
        # )
        # self.dataset_size = total_size

        self.data_dim = data_params.get("data_dim", 256)

        max_length = data_params.get("max_length", 6)
        min_length = data_params.get("min_length", 2)

        self.set_lengths = torch.randint(
            min_length, max_length, (self.dataset_size, )
        )

        cost_metric = data_params.get("cost_metric", "p-norm")
        cost_metric_args = list(
            data_params.get("cost_metric_args", {
                "p": 2
            }).values()
        )
        self.get_cost_matrix = METRIC_FUNCTION_FACTORY[cost_metric](
            *cost_metric_args
        )

        self.dataset = torch.zeros(self.dataset_size, self.data_dim)

        synthetic_dataset = SyntheticDataset(
            self.seed, data_params.get("distribution_type", "normal"),
            data_params.get("distribution_args", {
                "loc": 0,
                "scale": 1
            }), self.data_dim, self.dataset_size
        )

        self.set1_dataset = synthetic_dataset.__getitem__(max_length)

        self.permute = eval(data_params.get("permute", "False"))

        if self.permute:
            self.permutation_indices = list(
                map(torch.randperm, self.set_lengths)
            )
        else:
            self.set2_dataset = synthetic_dataset.__getitem__(max_length)

    def get_targets(self, dataset: torch.Tensor, index: int) -> Tuple:
        """Get one training sample.

        Args:
            dataset (torch.Tensor): Dummy tensor of zeros that facilitates
                sampling from a distribution on the fly.
            index (int): The index to be sampled.

        Returns:
            Tuple: Tuple containing sampled set1, sampled set2, hungarian
                matching indices of set1 vs set2 and set2 vs set1, and length
                of the sets.
        """

        # set1 = self.synthetic_dataset.__getitem__(
        #     index, self.set_lengths[index]
        # ).squeeze()

        set1 = self.set1_dataset[index, :self.set_lengths[index], :].squeeze()

        if self.permute is True:
            permute_idx = self.permutation_indices[index]
            set2 = set1[permute_idx, :]
        else:
            set2 = self.set2_dataset[index, :self.
                                     set_lengths[index], :].squeeze()
            # set2 = self.synthetic_dataset.__getitem__(
            #     index, self.set_lengths[index]
            # ).squeeze()

        cost_matrix = self.get_cost_matrix(set1, set2)

        matrix = torch.zeros_like(cost_matrix)
        rows, cols = linear_sum_assignment(cost_matrix)
        matrix[rows, cols] = 1
        idx12 = torch.from_numpy(cols)

        idx21 = torch.nonzero(matrix.t(), as_tuple=True)[1]

        return set1, set2, idx12, idx21, set1.size(0)

    def __len__(self) -> int:
        """Returns length of dataset"""
        return self.dataset_size

    def __getitem__(self, index: int) -> Tuple:
        """Generates one sample from the dataset.

        Args:
            index (int): The index to be sampled.

        Returns:
            Tuple : Tuple containing sampled set1, sampled set2, hungarian
                matching indices of set1 vs set2 and set2 vs set1, and length
                of the sets.
        """

        return self.get_targets(self.dataset[index], index)


class CollatorSetMatching:
    """Contains function to pad data returned by dataloader."""

    def __init__(self, dim: int, max_length: int, batch_first: bool = True):
        """Constructor.

        Args:
            dim (int): Dimension of the data.
            max_length (int): Maximum set length.
            batch_first (bool, optional): Whether batch size is the first
                dimension or not. Defaults to True.
        """
        super(CollatorSetMatching, self).__init__()
        self.dim = dim
        self.max_len = max_length - 1
        self.batch_first = batch_first

    def __call__(self, DataLoaderBatch: Tuple) -> Tuple:
        """Collate function for batch-wise padding of samples.

        Args:
            DataLoaderBatch (Tuple): Tuple of tensors returned by get_item of the
            dataset class.

        Returns:
            Tuple: Tuple of padded input tensors and tensor of set lengths.
                Note: For Seq2Seq model, remove connecting token and
                return ps1 instead of ps1_ct.
        """

        batch_size = len(DataLoaderBatch)
        batch_split = list(zip(*DataLoaderBatch))

        sets1, sets2, targs12, targs21, lengths = batch_split[0], batch_split[
            1], batch_split[2], batch_split[3], batch_split[4]

        pad_token = 10

        padded_sets1 = np.full(
            (batch_size, self.max_len, self.dim), pad_token, dtype=np.float32
        )
        padded_sets2 = np.full(
            (batch_size, self.max_len, self.dim), pad_token, dtype=np.float32
        )
        targets12 = np.tile(np.arange(self.max_len), (batch_size, 1))
        targets21 = np.tile(np.arange(self.max_len), (batch_size, 1))

        for i, l in enumerate(lengths):
            padded_sets1[i, 0:l, :] = sets1[i][0:l, :]
            padded_sets2[i, 0:l, :] = sets2[i][0:l, :]

            targets12[i, 0:l] = targs12[i][:]
            targets21[i, 0:l] = targs21[i][:]

        targets12 = torch.tensor(targets12)
        targets21 = torch.tensor(targets21)
        set_lens = torch.tensor(lengths)

        ps1 = torch.tensor(padded_sets1)

        ps2 = torch.tensor(padded_sets2)

        if self.batch_first is False:
            ps1, ps2 = ps1.permute(1, 0, 2), ps2.permute(1, 0, 2)

        return ps1, ps2, targets12, targets21, set_lens
