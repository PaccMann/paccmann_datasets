"""Implementation of _CsvEagerDataset."""
import numpy as np
import pandas as pd
from collections import OrderedDict
from ._csv_dataset import _CsvDataset
from ..types import FeatureList


class _CsvEagerDataset(_CsvDataset):
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
        _CsvDataset.__init__(
            self, filepath, feature_list=feature_list, **kwargs
        )

    def setup_dataset(self) -> None:
        """Setup the dataset."""
        self.df = self.feature_fn(pd.read_csv(self.filepath, **self.kwargs))
        self.min_max_scaler.fit(self.df.values)
        self.standardizer.fit(self.df.values)
        self.sample_to_index_mapping = {
            sample: index
            for index, sample in enumerate(self.df.index.tolist())
        }
        self.index_to_sample_mapping = {
            index: sample
            for sample, index in self.sample_to_index_mapping.items()
        }
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

    def __len__(self) -> int:
        """Total number of samples."""
        return self.df.shape[0]

    def __getitem__(self, index: int) -> np.array:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            torch.tensor: a torch tensor of token indexes,
                for the current sample.
        """
        return self.df.iloc[index].values
