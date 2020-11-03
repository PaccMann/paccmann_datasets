"""Implementation of _FastaEagerDataset."""
from upfp import parse_fasta

from ..types import Hashable
from .base_dataset import KeyDataset


class _FastaEagerDataset(KeyDataset):
    """
    .fasta dataset using eager loading.

    Suggested when handling datasets that can fit in the device memory.

    Only supports uniprot fasta headers
    """

    def __init__(
        self, fasta_filepath: str, gzipped=True, name: str = 'Sequence'
    ) -> None:
        """
        Initialize a .fasta dataset.

        Args:
            fasta_filepath (str): path to .fasta file.
            gzipped (bool): Whether or not fasta file is zipped (.fasta.gz).
            name (str): type of dataset, used to index columns.
        """
        KeyDataset.__init__(self)
        self.fasta_filepath = fasta_filepath
        self.name = name

        self.key_to_index_mapping = {}
        self.ordered_keys = []
        self.samples = []
        try:
            for index, item in enumerate(parse_fasta(fasta_filepath, gzipped=gzipped)):
                key = item['accession_number']  # uniprot unique identifier
                self.key_to_index_mapping[key] = index
                self.ordered_keys.append(key)
                self.samples.append(item['sequence'])
        except KeyError:
            raise KeyError('Badly formatted .fasta file, no sequence found.')

    def __len__(self) -> int:
        """Total number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> str:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            str: a sequence in the FASTA file.
        """
        return self.samples[index]

    def get_key(self, index: int) -> Hashable:
        """Get key from integer index."""
        return self.ordered_keys[index]

    def get_index(self, key: Hashable) -> int:
        """Get index for first datum mapping to the given key."""
        return self.key_to_index_mapping[key]

    @property
    def has_duplicate_keys(self):
        return len(self.ordered_keys) != len(self.key_to_index_mapping)

    def keys(self):
        return iter(self.ordered_keys)
