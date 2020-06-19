"""Implementation of _FastaLazyDataset."""
from pyfaidx import Fasta

from ..types import Hashable, Callable, Iterator
from .base_dataset import KeyDataset, DatasetDelegator


class _FastaLazyDataset(KeyDataset):
    """
    .fasta dataset using lazy loading via pyfaidx, which creates a small flat
    index file ".fai".

    Suggested when handling datasets that can fit in the device memory.

    Only supports uniprot fasta headers
    """

    def __init__(
        self, fasta_filepath: str,
        name: str = 'Sequence',
        key_function: Callable[str, str] = lambda x: x,
        **kwargs

    ) -> None:
        """[summary]

        Args:
            fasta_filepath (str): path to .fasta file.
            name (str, optional): type of dataset. Defaults to 'Sequence'.
            key_function (Callable[str, str], optional): function returning a
                unique key given the FASTA sequence header. Defaults to
                identity.
            kwargs (dict): Additional parameters passed to pyfaidx.Fasta class.

        Raises:
            ValueError: in case of duplicate keys from key_function in dataset.

        """
        self.fasta_filepath = fasta_filepath
        self.name = name
        self.datasource = Fasta(
            filename=fasta_filepath,
            key_function=key_function,
            as_raw=False,
            **kwargs
        )
        DatasetDelegator.__init__(self)
        KeyDataset.__init__(self)

        self.key_to_index_mapping = {
            key: index for index, key in enumerate(self.datasource.keys())
        }

    def __getitem__(self, index: int) -> str:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            str: a sequence in the FASTA file.
        """
        return str(self.datasource[index])

    def get_item_from_key(self, key: Hashable) -> str:
        """Get item via sample identifier"""
        return str(self.datasource[key])

    def get_key(self, index: int) -> Hashable:
        """Get sample identifier from integer index."""
        return self.datasource[index].name

    def get_index(self, key: Hashable) -> int:
        """Get index for first datum mapping to the given sample identifier."""
        return self.key_to_index_mapping[key]

    def keys(self) -> Iterator:
        """Default iterator of keys by iterating over dataset indices."""
        return self.datasource.keys()

    @property
    def has_duplicate_keys(self):
        # would error on init
        return False
