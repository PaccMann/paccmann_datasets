"""Implementation of _SmiEagerDataset."""
from torch.utils.data import Dataset
from .base_dataset import IndexedDataset
from .dataframe_dataset import DataFrameDataset
from ..files import read_smi
from ..types import Any, Hashable


class _SmiEagerDataset(DataFrameDataset):
    """
    .smi dataset using eager loading.

    Suggested when handling datasets that can fit in the device memory.
    In case of out of memory errors consider using _SmiLazyDataset.
    """

    def __init__(
        self, smi_filepath: str, name: str = 'SMILES', **kwargs
    ) -> None:
        """
        Initialize a .smi dataset.

        Args:
            smi_filepath (str): path to .smi file.
            name (str): type of dataset, used to index columns.
        """
        self.smi_filepath = smi_filepath
        self.name = name
        df = read_smi(self.smi_filepath, names=[self.name])
        DataFrameDataset.__init__(self, df)

    def __getitem__(self, index: int) -> str:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            str: SMILES for the current sample.
        """
        return self.df.iloc[index][self.name]

    def get_item_from_key(self, key: Hashable) -> Any:
        """Get item via sample identifier"""
        return self.df.loc[key, self.name]
