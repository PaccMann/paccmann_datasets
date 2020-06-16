"""Implementation of _CsvEagerDataset."""
import pandas as pd
from collections import OrderedDict
from ._csv_statistics import _CsvStatistics
from .dataframe_dataset import DataFrameDataset
from ..types import FeatureList


class _CsvEagerDataset(_CsvStatistics, DataFrameDataset):
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
                Except from nrows.
        """
        _CsvStatistics.__init__(
            self, filepath, feature_list=feature_list, **kwargs
        )  # calls setup_datasource

    def setup_datasource(self) -> None:
        """Setup the datasource ready to collect statistics."""
        df = self.feature_fn(pd.read_csv(self.filepath, **self.kwargs))
        # KeyDataset implementation, sets self.df
        DataFrameDataset.__init__(self, df)
        self.min_max_scaler.fit(self.df.values)
        self.standardizer.fit(self.df.values)
        self.feature_list = self.df.columns.tolist()
        self.feature_mapping = pd.Series(
            OrderedDict(
                [
                    (feature, index)
                    for index, feature in enumerate(self.feature_list)
                ]
            )
        )
        self.feature_fn = lambda df: df[self.feature_list]
