"""Implementation of _FastaEagerDataset."""
from torch.utils.data import Dataset
from upfp import parse_fasta


class _FastaEagerDataset(Dataset):
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
        Dataset.__init__(self)
        self.fasta_filepath = fasta_filepath
        self.name = name

        try:
            self.index_to_sample_mapping = dict(
                zip(
                    *list(
                        zip(
                            *[
                                (index, item['sequence'])
                                for index, item in enumerate(
                                    parse_fasta(
                                        fasta_filepath, gzipped=gzipped
                                    )
                                )
                            ]
                        )
                    )
                )
            )

        except KeyError:
            raise KeyError('Badly formatted .fasta file, no sequence found.')

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
        return self.index_to_sample_mapping[index]
