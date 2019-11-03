"""Implementation of _SmiLazyDataset."""
from ._cache_dataset import _CacheDataset
from ..files import read_smi


class _SmiLazyDataset(_CacheDataset):
    """
    .smi dataset using lazy loading.

    Suggested when handling datasets that can't fit in the device memory.
    In case of datasets fitting in device memory consider using
    _SmiEagerDataset for better performance.
    """

    def __init__(self, smi_filepath: str, chunk_size: int = 10000) -> None:
        """
        Initialize a .smi dataset.

        Args:
            smi_filepath (str): path to .smi file.
            chunk_size (int): size of the chunks. Defauls to 10000.
        """
        super(_SmiLazyDataset, self).__init__()
        self.smi_filepath = smi_filepath
        self.chunk_size = chunk_size
        self.sample_to_index_mapping = {}
        index = 0
        for chunk in read_smi(self.smi_filepath, chunk_size=self.chunk_size):
            for row_index, row in chunk.iterrows():
                self.cache[index] = row['SMILES']
                self.sample_to_index_mapping[row_index] = index
                index += 1
        self.index_to_sample_mapping = {
            index: sample
            for sample, index in self.sample_to_index_mapping.items()
        }
        self.number_of_samples = len(self.sample_to_index_mapping)

    def __len__(self) -> int:
        """Total number of samples."""
        return self.number_of_samples

    def __getitem__(self, index: int) -> str:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            torch.tensor: a torch tensor of token indexes,
                for the current sample.
        """
        return self.cache[index]
