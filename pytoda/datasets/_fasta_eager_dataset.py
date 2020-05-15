"""Implementation of _FastaEagerDataset."""
from ..base_dataset import IndexedDataset
from upfp import parse_fasta
from ..types import Hashable


class _FastaEagerDataset(IndexedDataset):  # base_dataset: needs test
    """
    .fasta dataset using eager loading.

    Suggested when handling datasets that can fit in the device memory.
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
        IndexedDataset.__init__(self)
        self.fasta_filepath = fasta_filepath
        self.name = name

        self.key_to_index_mapping = {}
        self.ordered_keys = []
        self.samples = []
        try:
            for index, item in enumerate(
                    parse_fasta(fasta_filepath, gzipped=gzipped)
            ):
                key = item['accession_number']  # uniprot unique identifier
                self.key_to_index_mapping[key] = index
                self.ordered_keys.append(key)
                self.samples.append(item['sequence'])
        except KeyError:
            raise KeyError('Badly formatted .fasta file, no sequence found.')   # base_dataset: needs test, only supports uniprot fasta

    def __len__(self) -> int:
        """Total number of samples."""
        return len(self.index_to_sample_mapping)

    def __getitem__(self, index: int) -> str:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            torch.tensor: a torch tensor of token indexes,
                for the current sample.
        """
        return self.samples[index]

    def get_key(self, index: int) -> Hashable:
        """Get sample identifier from integer index."""
        return self.ordered_keys[index]

    def get_index(self, key: Hashable) -> int:
        """Get index for first datum mapping to the given sample identifier."""
        return self.key_to_index_mapping[key]

    def keys(self):
        return iter(self.ordered_keys)
