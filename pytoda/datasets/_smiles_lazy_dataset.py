"""Implementation of _SMILESLazyDataset."""
import torch
from ..types import FileList
from ..smiles.smiles_language import SMILESLanguage
from ._smiles_dataset import _SMILESDataset
from ._smi_lazy_dataset import _SmiLazyDataset
from .utils import concatenate_file_based_datasets


class _SMILESLazyDataset(_SMILESDataset):
    """
    SMILES dataset using lazy loading.

    Suggested when handling datasets that can't fit in the device memory.
    In case of datasets fitting in device memory consider using
    _SMILESEagerDataset for better performance.
    """

    def __init__(
        self,
        smi_filepaths: FileList,
        smiles_language: SMILESLanguage = None,
        padding: bool = True,
        padding_length: int = None,
        add_start_and_stop: bool = False,
        augment: bool = False,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu'),
        chunk_size: int = 10000
    ) -> None:
        """
        Initialize a SMILES lazy dataset.

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
            chunk_size (int): size of the chunks. Defauls to 10000.
        """
        self.chunk_size = chunk_size
        super(_SMILESLazyDataset, self).__init__(
            smi_filepaths=smi_filepaths,
            smiles_language=smiles_language,
            padding=padding,
            add_start_and_stop=add_start_and_stop,
            augment=augment,
            device=device
        )

    def _setup_dataset(self) -> None:
        """Setup the dataset."""
        self._dataset = concatenate_file_based_datasets(
            filepaths=self.smi_filepaths,
            dataset_class=_SmiLazyDataset,
            chunk_size=self.chunk_size
        )
