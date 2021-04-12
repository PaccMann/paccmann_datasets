"""Implementation of _SmiEagerDataset."""
from ..files import read_smi
from ..types import Any, Hashable, Sequence
from .dataframe_dataset import DataFrameDataset


class _SmiEagerDataset(DataFrameDataset):
    """
    .smi dataset using eager loading.

    Suggested when handling datasets that can fit in the device memory.
    In case of out of memory errors consider using _SmiLazyDataset.
    """

    def __init__(
        self,
        smi_filepath: str,
        index_col: int = 1,
        name: str = 'SMILES',
        names: Sequence[str] = None,
    ) -> None:
        """
        Initialize a .smi dataset.

        Args:
            smi_filepath (str): path to .smi file.
            index_col (int): Data column used for indexing, defaults to 1.
            name (str): type of dataset, used to index columns in smi, and must
                be in names. Defaults to 'SMILES'.
            names (Sequence[str]): User-assigned names given to the columns.
                Defaults to `[name]`.
        """
        self.smi_filepath = smi_filepath
        self.name = name
        self.names = names or [name]
        self.index_col = index_col
        df = read_smi(self.smi_filepath, index_col=self.index_col, names=self.names)
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
        """Get item via key"""
        return self.df.loc[key, self.name]
