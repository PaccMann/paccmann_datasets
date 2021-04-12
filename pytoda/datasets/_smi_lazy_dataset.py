"""Implementation of _SmiLazyDataset."""
from ..files import read_smi
from ..types import Hashable, Sequence
from ._cache_datasource import _CacheDatasource
from .base_dataset import KeyDataset


class _SmiLazyDataset(KeyDataset, _CacheDatasource):
    """
    .smi dataset using lazy loading.

    Suggested when handling datasets that can't fit in the device memory.
    In case of datasets fitting in device memory consider using
    _SmiEagerDataset for better performance.
    """

    def __init__(
        self,
        smi_filepath: str,
        index_col: int = 1,
        chunk_size: int = 10000,
        name: str = 'SMILES',
        names: Sequence[str] = None,
    ) -> None:
        """
        Initialize a .smi dataset.

        Args:
            smi_filepath (str): path to .smi file.
            index_col (int): Data column used for indexing, defaults to 1.
            chunk_size (int): size of the chunks. Defaults to 10000.
            name (str): type of dataset, used to index columns in smi, and must
                be in names. Defaults to 'SMILES'.
            names (Sequence[str]): User-assigned names given to the columns.
                Defaults to `[name]`.
        """
        _CacheDatasource.__init__(self, fit_size_limit_filepath=smi_filepath)
        KeyDataset.__init__(self)
        self.smi_filepath = smi_filepath
        self.name = name
        self.names = names or [name]
        self.index_col = index_col
        self.chunk_size = chunk_size
        self.key_to_index_mapping = {}
        index = 0
        self.ordered_keys = []
        for chunk in read_smi(
            self.smi_filepath,
            index_col=self.index_col,
            chunk_size=self.chunk_size,
            names=self.names,
        ):
            for key, row in chunk.iterrows():
                self.cache[index] = row[self.name]
                self.key_to_index_mapping[key] = index
                index += 1
                self.ordered_keys.append(key)

        self.number_of_samples = len(self.ordered_keys)

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
        """Delete the _SmiLazyDataset."""
        _CacheDatasource.__del__(self)
