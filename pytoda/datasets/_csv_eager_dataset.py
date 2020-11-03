"""Implementation of _CsvEagerDataset."""
from collections import OrderedDict

import pandas as pd

from ..types import CallableOnSource, FeatureList, Optional
from ._csv_statistics import _CsvStatistics
from .dataframe_dataset import DataFrameDataset


class _CsvEagerDataset(DataFrameDataset, _CsvStatistics):
    """
    .csv dataset using eager loading.

    Suggested when handling datasets that can fit in the device memory.
    In case of out of memory errors consider using _CsvLazyDataset.
    """

    def __init__(
        self, filepath: str, feature_list: FeatureList = None, **kwargs
    ) -> None:
        """
        Initialize a .csv dataset.

        Args:
            filepath (str): path to .csv file.
            feature_list (FeatureList): a list of features. Defaults to None.
            kwargs (dict): additional parameters for pd.read_csv.
        """
        _CsvStatistics.__init__(
            self, filepath, feature_list=feature_list, **kwargs
        )  # calls setup_datasource

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
        feature_mapping (pd.Series): maps feature name to index in items.
        feature_fn (CallableOnSource): function that indexes datasource with
            the feature_list.
        """
        df = self.preprocess_df(pd.read_csv(self.filepath, **self.kwargs))
        # KeyDataset implementation, sets self.df
        DataFrameDataset.__init__(self, df)
        self.min_max_scaler.fit(self.df.values)
        self.standardizer.fit(self.df.values)
        self.notna_count = self.df.notna().sum().values

        self.feature_list = self.df.columns.tolist()
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

        def feature_fn(df: pd.DataFrame) -> pd.DataFrame:
            return df[feature_list]

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
        self.df = transform_fn(feature_fn(self.df))
        if impute is not None:
            self.df = self.df.fillna(impute)
