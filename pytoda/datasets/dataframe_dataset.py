"""KeyDataset for pandas DataFrames ."""
import numpy as np
import pandas as pd

from ..types import Hashable, Iterator
from .base_dataset import KeyDataset


class DataFrameDataset(KeyDataset):
    """
    Dataset of rows from pandas.DataFrame
    """

    def __init__(self, df: pd.DataFrame):
        super(DataFrameDataset).__init__()
        self.df = df

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
        """Get key from integer index."""
        return self.df.index[index]

    def get_index(self, key: Hashable) -> int:
        """Get index for first datum mapping to the given key."""
        # item will raise if not single value (deprecated in pandas)
        try:
            indexes = np.nonzero(self.df.index == key)[0]
            return indexes.item()
        except ValueError:
            if len(indexes) == 0:
                raise KeyError
            else:
                # key not unique, return first as ConcatKeyDataset
                return indexes[0]

    def get_item_from_key(self, key: Hashable) -> np.array:
        """Get item via key"""
        return self.df.loc[key, :].values

    def keys(self) -> Iterator:
        """Iterator over index of dataframe."""
        return iter(self.df.index)
