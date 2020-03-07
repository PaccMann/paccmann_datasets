"""Implementation of _SmiEagerDataset."""
from torch.utils.data import Dataset
from ..files import read_smi


class _SmiEagerDataset(Dataset):
    """
    .smi dataset using eager loading.

    Suggested when handling datasets that can fit in the device memory.
    In case of out of memory errors consider using _SmiLazyDataset.
    """

    def __init__(self, smi_filepath: str, name: str = 'SMILES') -> None:
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
            for index, sample in enumerate(self.df.index.tolist())
        }
        self.index_to_sample_mapping = {
            index: sample
            for sample, index in self.sample_to_index_mapping.items()
        }

    def __len__(self) -> int:
        """Total number of samples."""
        return self.df.shape[0]

    def __getitem__(self, index: int) -> str:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            torch.tensor: a torch tensor of token indexes,
                for the current sample.
        """
        return self.df.iloc[index][self.name]
