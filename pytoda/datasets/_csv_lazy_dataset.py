"""Implementation of _CsvLazyDataset."""
import os
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

from ..types import FeatureList
from ._cache_dataset import _CacheDataset
from ._csv_dataset import _CsvDataset


def sizeof_fmt(num, suffix='B'):
    """Source: https://stackoverflow.com/a/1094933"""
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


class _CsvLazyDataset(_CacheDataset, _CsvDataset):
    """
    .csv dataset using lazy loading.

    Suggested when handling datasets that can't fit in the device memory.
    In case of datasets fitting in device memory consider using
    _CsvEagerDataset for better performance.
    """

    def __init__(
        self,
        filepath: str,
        feature_list: FeatureList = None,
        chunk_size: int = 10000,
        **kwargs
    ) -> None:
        """
        Initialize a .csv dataset.

        Args:
            filepath (str): path to .csv file.
            feature_list (FeatureList): a list of features. Defaults to None.
            chunk_size (int): size of the chunks. Defauls to 10000.
            kwargs (dict): additional parameters for pd.read_csv.
                Except from nrows and chunksize.
        """
        self.chunk_size = chunk_size
        size_limit = 1073741824  # default limit of 1GiB from diskcash
        file_size = os.path.getsize(filepath)
        if file_size > size_limit:
            size_limit = file_size
            message = (
                f'Temporary directory for caching can be up to {size_limit} '
                f'bytes ({sizeof_fmt(size_limit)}) large to fit data.'
            )
            # ResourceWarning is usually filtered by default
            warnings.warn(message, ResourceWarning)
        _CacheDataset.__init__(self, size_limit=size_limit)
        _CsvDataset.__init__(
            self, filepath, feature_list=feature_list, **kwargs
        )
        # NOTE: make sure chunksize is not passed twice
        _ = self.kwargs.pop('chunksize', None)

    def setup_dataset(self) -> None:
        """Setup the dataset."""
        self.sample_to_index_mapping = {}
        index = 0
        for chunk in pd.read_csv(
            self.filepath, chunksize=self.chunk_size, **self.kwargs
        ):
            chunk = self.feature_fn(chunk)
            self.min_max_scaler.partial_fit(chunk.values)
            self.standardizer.partial_fit(chunk.values)
            for row_index, row in chunk.iterrows():
                self.cache[index] = row.values
                self.sample_to_index_mapping[row_index] = index
                index += 1
        self.index_to_sample_mapping = {
            index: sample
            for sample, index in self.sample_to_index_mapping.items()
        }
        self.number_of_samples = len(self.sample_to_index_mapping)
        self.feature_list = chunk.columns.tolist()
        self.feature_mapping = pd.Series(
            OrderedDict(
                [
                    (feature, index)
                    for index, feature in enumerate(self.feature_list)
                ]
            )
        )
        self.feature_fn = lambda sample: sample[self.feature_mapping[
            self.feature_list].values]

    def __len__(self) -> int:
        """Total number of samples."""
        return self.number_of_samples

    def __getitem__(self, index: int) -> np.array:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            torch.tensor: a torch tensor of token indexes,
                for the current sample.
        """
        return self.cache[index]

    def __del__(self):
        """Delete the _CsvLazyDataset."""
        _CacheDataset.__del__(self)
