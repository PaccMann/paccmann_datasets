"""IndexedDataset for pandas DataFrames ."""
import numpy as np
import pandas as pd
from .base_dataset import IndexedDataset
from ..types import Hashable, Iterator


class DataFrameDataset(IndexedDataset):
    """
    Dataset from pandas.DataFrame
    """
    def __init__(self, df: pd.DataFrame):
        super(DataFrameDataset).__init__()
        self.df = df
        self._range_index = pd.RangeIndex(0, self.__len__())

    def __len__(self) -> int:
        """Total number of samples."""
        return len(self.df)

    def __getitem__(self, index: int) -> np.array:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            np.array: a selected row of the CSV.
        """
        return self.df.iloc[index].values

    def get_key(self, index: int) -> Hashable:
        """Get sample identifier from integer index."""
        return self.df.index[index]

    def get_index(self, key: Hashable) -> int:
        """Get index for first datum mapping to the given sample identifier."""
        # item will raise if not single value (deprecated in pandas)
        try:
            index = self._range_index[
                self.datasource.index == key
            ]
            return index.values.item()
        except ValueError:
            if len(index) == 0:
                raise KeyError
            else:
                # key not unique, return first as _ConcatenatedDataset
                return index.values[0]

    def get_item_from_key(self, key: Hashable) -> np.array:
        """Get item via sample identifier"""
        return self.df.loc[key, :].values

    def keys(self) -> Iterator:
        """Iterator over index of dataframe."""
        return iter(self.df.index)
