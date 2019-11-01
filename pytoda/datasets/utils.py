"""Utils for the dataset module."""
from torch.utils.data import Dataset
from ._concatenated_dataset import _ConcatenatedDataset
from ..types import FileList


def concatenate_file_based_datasets(
    filepaths: FileList, dataset_class: type, **kwargs
) -> Dataset:
    """
    Concatenate file-based datasets into a single one.

    Args:
        filepaths (FileList): list of filepaths.
        dataset_class (type): dataset class.
        kwargs (dict): additional arguments for
            dataset_class.__init__(self, filepath, **kwargs).
    Returns:
        Dataset: the concatenated dataset.
    """
    return _ConcatenatedDataset(
        datasets=[dataset_class(filepath, **kwargs) for filepath in filepaths]
    )
