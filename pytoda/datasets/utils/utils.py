"""Utils for the dataset module."""
from copy import copy

from ...types import Any, Files, Hashable, List, Tuple
from ..base_dataset import AnyBaseDataset, ConcatKeyDataset
from .factories import BACKGROUND_TENSOR_FACTORY


def sizeof_fmt(num, suffix='B'):
    """
    Human readable file size. Source: https://stackoverflow.com/a/1094933
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def concatenate_file_based_datasets(
    filepaths: Files, dataset_class: type, **kwargs
) -> ConcatKeyDataset:
    r"""
    Concatenate file-based datasets into a single one, with the ability to
        get the source dataset of items.

    Args:
        filepaths (Files): list of filepaths.
        dataset_class (type): dataset class reading from file.
            Supports KeyDataset and DatasetDelegator.
            For pure torch.utils.data.Dataset the returned instance can
            still be used like a `pytoda.datasets.TransparentConcatDataset`,
            but methods depending on key lookup will fail.
        kwargs (dict): additional args for dataset_class.__init__(filepath, \**kwargs).

    Returns:
        ConcatKeyDataset: the concatenated dataset.
    """

    return ConcatKeyDataset(
        datasets=[dataset_class(filepath, **kwargs) for filepath in filepaths]
    )


def indexed(dataset: AnyBaseDataset) -> AnyBaseDataset:
    """
    Returns mutated shallow copy of passed dataset instance, where indexing
    behavior is changed to additionally returning index.
    """
    default_getitem = dataset.__getitem__  # bound method
    default_from_key = dataset.get_item_from_key  # bound method

    def return_item_index_tuple(self, index: int) -> Tuple[Any, int]:
        return_index = len(self) + index if index < 0 else index
        return default_getitem(index), return_index

    def return_item_index_tuple_from_key(self, key: Hashable) -> Tuple[Any, int]:
        """prevents `get_item_from_key` to call new indexed __getitem__"""
        return default_from_key(key), dataset.get_index(key)

    methods = {
        '__getitem__': return_item_index_tuple,
        'get_item_from_key': return_item_index_tuple_from_key,
    }
    ds = copy(dataset)
    ds.__class__ = type(
        f'Indexed{type(dataset).__name__}', (dataset.__class__,), methods
    )
    return ds


def keyed(dataset: AnyBaseDataset) -> AnyBaseDataset:
    """
    Returns mutated shallow copy of passed dataset instance, where indexing
    behavior is changed to additionally returning key.
    """
    default_getitem = dataset.__getitem__  # bound method
    default_from_key = dataset.get_item_from_key  # bound method

    def return_item_key_tuple(self, index: int) -> Tuple[Any, Hashable]:
        return (default_getitem(index), dataset.get_key(index))

    def return_item_key_tuple_from_key(self, key: Hashable) -> Tuple[Any, Hashable]:
        """prevents `get_item_from_key` to call new keyed __getitem__"""
        return default_from_key(key), key

    methods = {
        '__getitem__': return_item_key_tuple,
        'get_item_from_key': return_item_key_tuple_from_key,
    }
    ds = copy(dataset)
    ds.__class__ = type(f'Keyed{type(dataset).__name__}', (dataset.__class__,), methods)
    return ds


def pad_item(
    item: Tuple,
    padding_modes: List[str],
    padding_values: List,
    max_length: int,
) -> Tuple:
    """Padding function for a single item of a batch.

    Args:
        item (Tuple): Tuple returned by the __getitem__ function of a Dataset class.
        padding_modes (List[str]): The type of padding to perform for each datum in item.
            Options are 'constant' for constant value padding, and 'range' to fill the
            tensor with a range of values.
        padding_values (List): The values with which to fill the background tensor for padding.
            Can be a constant value or a range depending on the datum to pad in item.
        max_length (int): The maximum length to which the datum should be padded.

    Returns:
        Tuple: Tuple of tensors padded according to the given specifications.

    NOTE: pad_item function uses trailing dimensions as the repetitions argument
          for range_tensor(), since the 'length' of the set is covered by the
          value_range. That is, if a tensor of shape (5,) is required for
          padding_mode 'range' then () is passed as shape into range_tensor
          function which will repeat range(5) exactly once thus giving us a (5,) tensor.
    """
    # for each Tensor in the list we determine the output dimensions
    max_sizes = [datum.size() for datum in item]
    out_dimses = [
        (max_length, *max_sizes[i][1:])
        if padding_modes[i] == 'constant'
        else (*max_sizes[i][1:],)
        for i in range(len(max_sizes))
    ]

    out_tensors = [
        BACKGROUND_TENSOR_FACTORY[mode](value, out_dims)
        for out_dims, mode, value in zip(out_dimses, padding_modes, padding_values)
    ]

    for datum_index, tensor in enumerate(item):
        length = tensor.size(0)
        out_tensors[datum_index][:length, ...] = tensor
    return out_tensors
