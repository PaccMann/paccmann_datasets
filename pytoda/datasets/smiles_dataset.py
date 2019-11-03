"""SMILESDataset module."""
import torch
from torch.utils.data import Dataset
from ..smiles.smiles_language import SMILESLanguage
from ._smiles_eager_dataset import _SMILESEagerDataset
from ._smiles_lazy_dataset import _SMILESLazyDataset
from ..types import FileList

SMILES_DATASET_IMPLEMENTATIONS = {
    'eager': _SMILESEagerDataset,
    'lazy': _SMILESLazyDataset
}


class SMILESDataset(Dataset):
    """
    SMILES dataset implementation.
    """

    def __init__(
        self,
        *smi_filepaths: FileList,
        smiles_language: SMILESLanguage = None,
        padding: bool = True,
        padding_length: int = None,
        add_start_and_stop: bool = False,
        augment: bool = False,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu'),
        backend: str = 'eager'
    ) -> None:
        """
        Initialize a SMILES dataset.

        Args:
            smi_filepaths (FileList): paths to .smi files.
            smiles_language (SMILESLanguage): a smiles language.
                Defaults to None.
            padding (bool): pad sequences to longest in the smiles language.
                Defaults to True.
            padding_length (int): manually sets number of applied paddings,
                applies only if padding is True. Defaults to None.
            add_start_and_stop (bool): add start and stop token indexes.
                Defaults to False.
            augment (bool): perform SMILES augmentation. Defaults to False.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
            backend (str): memeory management backend.
                Defaults to eager, prefer speed over memory consumption.
        """
        Dataset.__init__(self)
        if not (backend in SMILES_DATASET_IMPLEMENTATIONS):
            raise RuntimeError(
                'backend={} not supported! '.format(backend) +
                'Select one in [{}]'.
                format(','.join(SMILES_DATASET_IMPLEMENTATIONS.keys()))
            )
        self._dataset = SMILES_DATASET_IMPLEMENTATIONS[backend](
            smi_filepaths=smi_filepaths,
            smiles_language=smiles_language,
            padding=padding,
            add_start_and_stop=add_start_and_stop,
            augment=augment,
            device=device
        )
        self.smiles_language = self._dataset.smiles_language
        self.sample_to_index_mapping = self._dataset.sample_to_index_mapping

    def __len__(self) -> int:
        """Total number of samples."""
        return len(self._dataset)

    def __getitem__(self, index: int) -> torch.tensor:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            torch.tensor: a torch tensor of token indexes,
                for the current sample.
        """
        return self._dataset[index]
