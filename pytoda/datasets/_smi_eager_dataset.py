"""Implementation of _SmiEagerDataset."""
from torch.utils.data import Dataset
from .base_dataset import IndexedDataset
from ..files import read_smi
from ..types import Any, Hashable


class _SmiEagerDataset(IndexedDataset):
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
        Dataset.__init__(self)
        self.smi_filepath = smi_filepath
        self.name = name
        self.df = read_smi(self.smi_filepath, names=[self.name])
        self.sample_to_index_mapping = {
            sample: index
            for index, sample in enumerate(self.df.index)
        }

    def __len__(self) -> int:
        """Total number of samples."""
        return len(self.df)

    def __getitem__(self, index: int) -> str:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            str: SMILES for the current sample.
        """
        return self.df.iloc[index][self.name]

    def get_key(self, index: int) -> Hashable:
        """Get sample identifier from integer index."""
        return self.df.index[index]

    def get_index(self, key: Hashable) -> int:
        """Get integer index from sample identifier."""
        return self.sample_to_index_mapping[key]

    def get_item_from_key(self, key: Hashable) -> Any:
        """Get item via sample identifier"""
        return self.df.loc[key][self.name]

    def keys(self):
        return self.sample_to_index_mapping.keys()
