"""An extension of ConcatDataset with transparent indexing."""
import bisect
from torch.utils.data import ConcatDataset, Dataset
from typing import Tuple, List


class _ConcatenatedDataset(ConcatDataset):
    """_ConcatenatedDataset with transparent indexing."""

    def __init__(self, datasets: List[Dataset]):
        """
        Initialize the _ConcatenatedDataset.

        Args:
            datasets (List[Dataset]): a list of datasets.
        """
        super(_ConcatenatedDataset, self).__init__(datasets)

    def get_index_pair(self, idx: int) -> Tuple[int, int]:
        """
        Get dataset and sample indexes.

        Args:
            idx (int): index in the concatenated dataset.

        Returns:
            Tuple[int, int]: dataset and sample indexex.
        """
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length'
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return (dataset_idx, sample_idx)
