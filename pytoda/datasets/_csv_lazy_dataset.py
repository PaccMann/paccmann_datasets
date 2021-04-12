"""Implementation of _CsvLazyDataset."""
from collections import OrderedDict

import numpy as np
import pandas as pd

from ..types import CallableOnSource, FeatureList, Hashable, Optional
from ._cache_datasource import _CacheDatasource
from ._csv_statistics import _CsvStatistics
from .base_dataset import KeyDataset


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
        **kwargs,
    ) -> None:
        """
        Initialize a .csv dataset.

        Args:
            filepath (str): path to .csv file.
            feature_list (FeatureList): a list of features. Defaults to None.
            chunk_size (int): size of the chunks. Defaults to 10000.
            kwargs (dict): additional parameters for pd.read_csv.
                The argument chunksize is ignored in favor of chunck_size.
        """
        self.chunk_size = chunk_size

        _CacheDatasource.__init__(self, fit_size_limit_filepath=filepath)
        _CsvStatistics.__init__(
            self, filepath, feature_list=feature_list, **kwargs
        )  # calls setup_datasource
        _ = self.kwargs.pop('chunksize', None)  # not passing chunksize twice
        KeyDataset.__init__(self)

    def setup_datasource(self):
        """
        Setup the datasource and compute statistics.

        The dataframe is read, calling `self.preprocess_df` on it and setting
        up the data as source, collecting statistics of the data.

        NOTE:
        To read item with a different subset and order of features,
        use the function returned by `get_feature_fn` or use the
        feature_mapping to ensure integer indexing.

        Sets:
        feature_list (FeatureList): feature names in this datasource.
        feature_fn (CallableOnSource): function that indexes datasource with
            the feature_list.
        feature_mapping (pd.Series): maps feature name to index in items.
        """
        self.key_to_index_mapping = {}
        index = 0
        self.ordered_keys = []
        # look ahead for row length
        chunk = next(pd.read_csv(self.filepath, chunksize=1, **self.kwargs))
        self.notna_count = np.array([0] * self.preprocess_df(chunk).shape[1])

        for chunk in pd.read_csv(
            self.filepath, chunksize=self.chunk_size, **self.kwargs
        ):
            chunk = self.preprocess_df(chunk)
            self.min_max_scaler.partial_fit(chunk.values)
            self.standardizer.partial_fit(chunk.values)
            self.notna_count += chunk.notna().sum().values
            for key, row in chunk.iterrows():
                self.cache[index] = row.values
                self.key_to_index_mapping[key] = index
                index += 1
                self.ordered_keys.append(key)

        self.number_of_samples = len(self.ordered_keys)

        self.feature_list = chunk.columns.tolist()
        self.feature_mapping = pd.Series(
            OrderedDict(
                [(feature, index) for index, feature in enumerate(self.feature_list)]
            )
        )
        self.feature_fn = self.get_feature_fn(self.feature_list)

    def get_feature_fn(self, feature_list: FeatureList) -> CallableOnSource:
        """Provides datasource specific indexing.

        Args:
            feature_list (FeatureList): subset of features to return in order.

        Returns:
            CallableOnSource: function that indexes datasource with the
                feature_list.
        """
        indices = self.feature_mapping[feature_list].values

        def feature_fn(sample: np.ndarray) -> np.ndarray:
            return sample[indices]

        return feature_fn

    def transform_datasource(
        self,
        transform_fn: CallableOnSource,
        feature_fn: CallableOnSource,
        impute: Optional[float] = None,
    ) -> None:
        """Apply scaling to the datasource.

        Args:
            transform_fn (CallableOnSource): transformation on source data.
            feature_fn (CallableOnSource): function that indexes datasource.
            impute (Optional[float]): NaN imputation with value if
                given. Defaults to None.
        """
        if impute is None:
            for index in self.cache:
                self.cache[index] = transform_fn(feature_fn(self.cache[index]))
        else:
            for index in self.cache:
                transformed = transform_fn(feature_fn(self.cache[index]))
                self.cache[index] = np.nan_to_num(transformed, impute)

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
        """Get key from integer index."""
        return self.ordered_keys[index]

    def get_index(self, key: Hashable) -> int:
        """Get index for first datum mapping to the given key."""
        return self.key_to_index_mapping[key]

    def keys(self):
        return iter(self.ordered_keys)

    @property
    def has_duplicate_keys(self):
        return self.number_of_samples != len(self.key_to_index_mapping)

    def __del__(self):
        """Delete the _CsvLazyDataset."""
        _CacheDatasource.__del__(self)
