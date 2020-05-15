"""Implementation of _SmiLazyDataset."""
from ._cache_datasource import _CacheDatasource
from .base_dataset import IndexedDataset
from ..types import Hashable
from ..files import read_smi


class _SmiLazyDataset(IndexedDataset, _CacheDatasource):
    """
    .smi dataset using lazy loading.

    Suggested when handling datasets that can't fit in the device memory.
    In case of datasets fitting in device memory consider using
    _SmiEagerDataset for better performance.
    """

    def __init__(
        self, smi_filepath: str, name: str = 'SMILES', chunk_size: int = 10000
    ) -> None:
        """
        Initialize a .smi dataset.

        Args:
            smi_filepath (str): path to .smi file.
            chunk_size (int): size of the chunks. Defaults to 10000.
        """
        _CacheDatasource.__init__(self)
        IndexedDataset.__init__(self)
        self.smi_filepath = smi_filepath
        self.name = name
        self.chunk_size = chunk_size
        self.key_to_index_mapping = {}
        index = 0
        self.ordered_keys = []
        for chunk in read_smi(self.smi_filepath, chunk_size=self.chunk_size):
            for row_index, row in chunk.iterrows():
                self.cache[index] = row['SMILES']
                self.key_to_index_mapping[row_index] = index
                index += 1
                self.ordered_keys.append(row_index)

        self.number_of_samples = len(self.key_to_index_mapping)

    def __len__(self) -> int:
        """Total number of samples."""
        return self.number_of_samples

    def __getitem__(self, index: int) -> str:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            str: a SMILES for the current sample.
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

    def __del__(self):
        """Delete the _SmiLazyDataset."""
        _CacheDatasource.__del__(self)
