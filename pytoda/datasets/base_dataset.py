"""Implementation of base classes working with datasets."""
import bisect

import pandas as pd
from torch.utils.data import ConcatDataset, Dataset

from ..types import Any, Hashable, Iterator, List, Tuple, Union


class KeyDataset(Dataset):
    """
    Base Class for Datsets with both integer index and item identifier key.

    Implicit abstract methods are:
    `__len__(self)` https://github.com/pytorch/pytorch/blob/66a20c259b3b2063e59102ab23f3fb34fc819455/torch/utils/data/sampler.py#L23
    `__getitem__(self, index: int)` is inherited

    Default implementations to index using key and getting all keys are
    provided but should be overloaded when possible as calls to `get_item`
    and `get_key` might be expensive.

    The keys are expected to be unique. Call `has_duplicate_keys` to make sure.
    If there are duplicate keys, on lookup generally the first one found will
    be used, but there are no guarantees.
    """

    def __add__(self, other):
        return ConcatKeyDataset([self, other])

    def get_key(self, index: int) -> Hashable:
        """Get key from integer index."""
        raise NotImplementedError

    def get_index(self, key: Hashable) -> int:
        """Get index for first datum mapping to the given key."""
        raise NotImplementedError

    def get_item_from_key(self, key: Hashable) -> Any:
        """Get item via key"""
        return self.__getitem__(self.get_index(key))

    def keys(self) -> Iterator:
        """Default iterator of keys by iterating over dataset indexes."""
        for index in range(len(self)):
            yield self.get_key(index)

    @property
    def has_duplicate_keys(self) -> bool:
        """Check whether each key is unique."""
        return pd.Index(self.keys()).has_duplicates


class DatasetDelegator(Dataset):
    """
    Base class for KeyDataset attribute accesses from `self.dataset`.

    The attributes/methods to delegate are stored to allow explicit filtering
    and addition to class documentation.

    Source: https://www.fast.ai/2019/08/06/delegation/
    """

    # built-in methods need to be defined explicitly
    # https://stackoverflow.com/a/57589213
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]

    # returned dataset is not a Delegator
    def __add__(self, other):
        return ConcatKeyDataset([self, other])

    # explicit implementation to ensure use of top level __getitem__
    # (possibly overloaded) vs using datasets.__getitem__
    def get_item_from_key(self, key: Hashable) -> Any:
        """Get datum mapping to the given key."""
        return self.__getitem__(self.get_index(key))

    @staticmethod
    def _delegation_filter(method_name):
        """To remove unwanted attributes/methods from being delegated."

        Args:
            method_name (str): attribute/method of an instance.

        Returns:
            bool: wether to delegate method.
        """
        # do not delegate to all private attributes
        if method_name.startswith('_'):
            return False
        else:
            return True

    # call super().__getattribute__ is needed in place of self, as
    # calls to self could  lead to infinite loops, e.g. with copy

    @property
    def _delegatable(self):
        return [
            o
            for o in dir(
                super().__getattribute__('dataset')
            )  # other AttributeErrors are masked here if dataset is not set  # noqa
            if self._delegation_filter(o)
        ]

    # delegation, i.e. in case method not defined in class or class hierarchy
    def __getattr__(self, k):
        if k in super().__getattribute__('_delegatable'):
            return getattr(super().__getattribute__('dataset'), k)
        raise AttributeError(k)

    def __dir__(self):
        return dir(type(self)) + list(self.__dict__.keys()) + self._delegatable


class TransparentConcatDataset(ConcatDataset):
    """Extension of ConcatDataset with transparent indexing."""

    def __init__(self, datasets: List[Dataset]):
        super().__init__(datasets)

    def get_index_pair(self, idx: int) -> Tuple[int, int]:
        """
        Get dataset and sample indexes.

        Args:
            idx (int): index in the concatenated dataset.

        Returns:
            Tuple[int, int]: dataset and sample index.
        """
        # related request https://github.com/pytorch/pytorch/issues/32034
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


class ConcatKeyDataset(TransparentConcatDataset, KeyDataset):
    """
    Extension of ConcatDataset with transparent indexing supporting KeyDataset

    The keys are expected to be unique. If there are duplicate keys, on lookup
    the first one found will be used by default.
    """

    def __init__(self, datasets: List['AnyBaseDataset']):
        """
        Initialize the ConcatKeyDataset.

        Args:
            datasets (List[AnyBaseDataset]): a list of datasets.
        """
        super().__init__(datasets)
        # __getitem__ and __len__ implementation from ConcatDataset

    # the following require dataset to implement get_key and get_index
    def get_key_pair(self, index: int) -> Tuple[int, Hashable]:
        """Get dataset index key from integer index."""
        dataset_idx, sample_idx = self.get_index_pair(index)
        return dataset_idx, self.datasets[dataset_idx].get_key(sample_idx)

    def get_key(self, index: int) -> Hashable:
        """Get key from integer index."""
        return self.get_key_pair(index)[1]

    def get_index(self, key: Hashable) -> int:
        """Get index for first datum mapping to the given key."""
        for dataset_idx, dataset in enumerate(self.datasets):
            try:
                sample_idx = dataset.get_index(key)
            except KeyError:
                continue
            break
        else:
            raise KeyError(f'key {key} not found in datasets.')
        # return index from index_pair
        if dataset_idx == 0:
            return sample_idx
        else:
            return sample_idx + self.cumulative_sizes[dataset_idx - 1]

    def get_item_from_key(self, key: Hashable) -> Any:
        """Get datum mapping to the given key."""
        return self.__getitem__(self.get_index(key))

    def keys(self) -> Iterator:
        """Default generator of keys by iterating over dataset."""
        for dataset in self.datasets:
            for key in dataset.keys():
                yield key

    # default has_duplicate_keys via keys()


AnyBaseDataset = Union[KeyDataset, DatasetDelegator, ConcatKeyDataset]
