"""Utils for the dataset module."""
from ..types import FileList
from .base_dataset import ConcatKeyDataset


def concatenate_file_based_datasets(
    filepaths: FileList, dataset_class: type, **kwargs
) -> ConcatKeyDataset:
    """
    Concatenate file-based datasets into a single one, with the ability to
        get the source dataset of items.

    Args:
        filepaths (FileList): list of filepaths.
        dataset_class (type): dataset class reading from file.
            Supports KeyDataset and DatasetDelegator.
            For pure torch.utils.data.Dataset the returned instance can
            still be used like a `pytoda.datasets.TransparentConcatDataset`,
            but methods depending on key lookup will fail.
        kwargs (dict): additional arguments for
            dataset_class.__init__(self, filepath, **kwargs).
    Returns:
        ConcatKeyDataset: the concatenated dataset.
    """
    return ConcatKeyDataset(
        datasets=[dataset_class(filepath, **kwargs) for filepath in filepaths]
    )
