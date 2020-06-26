"""Implementation of _CsvLazyDataset."""
from collections import OrderedDict

import numpy as np
import pandas as pd
from ._cache_datasource import _CacheDatasource
from .base_dataset import KeyDataset
from ._csv_statistics import _CsvStatistics
from ..types import FeatureList, Hashable


class _CsvLazyDataset(KeyDataset, _CacheDatasource, _CsvStatistics):
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
            chunk_size (int): size of the chunks. Defaults to 10000.
            kwargs (dict): additional parameters for pd.read_csv.
                Except from nrows and chunksize.
        """
        self.chunk_size = chunk_size

        _CacheDatasource.__init__(self, fit_size_limit_filepath=filepath)
        _CsvStatistics.__init__(
            self, filepath, feature_list=feature_list, **kwargs
        )  # calls setup_datasource
        _ = self.kwargs.pop('chunksize', None)  # not passing chunksize twice
        KeyDataset.__init__(self)

    def setup_datasource(self) -> None:
        """Setup the datasource ready to collect statistics."""
        self.key_to_index_mapping = {}
        index = 0
        self.ordered_keys = []
        for chunk in pd.read_csv(
            self.filepath, chunksize=self.chunk_size, **self.kwargs
        ):
            chunk = self.feature_fn(chunk)
            self.min_max_scaler.partial_fit(chunk.values)
            self.standardizer.partial_fit(chunk.values)
            for key, row in chunk.iterrows():
                self.cache[index] = row.values
                self.key_to_index_mapping[key] = index
                index += 1
                self.ordered_keys.append(key)

        self.number_of_samples = len(self.ordered_keys)
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
            np.array: the current sample read from cache.
        """
        return self.cache[index]

    def get_key(self, index: int) -> Hashable:
        """Get sample identifier from integer index."""
        return self.ordered_keys[index]

    def get_index(self, key: Hashable) -> int:
        """Get index for first datum mapping to the given sample identifier."""
        return self.key_to_index_mapping[key]

    def keys(self):
        return iter(self.ordered_keys)

    @property
    def has_duplicate_keys(self):
        return self.number_of_samples != len(self.key_to_index_mapping)

    def __del__(self):
        """Delete the _CsvLazyDataset."""
        _CacheDatasource.__del__(self)
