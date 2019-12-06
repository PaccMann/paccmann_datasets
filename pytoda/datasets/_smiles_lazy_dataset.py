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
        canonical: bool = False,
        augment: bool = False,
        kekulize: bool = False,
        all_bonds_explicit: bool = False,
        all_hs_explicit: bool = False,
        randomize: bool = False,
        remove_bonddir: bool = False,
        remove_chirality: bool = False,
        selfies: bool = False,
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
            canonical (bool): performs canonicalization of SMILES (one original string for one molecule),
                if canonical=True, then other transformations (augment etc, see below) do not apply
            augment (bool): perform SMILES augmentation. Defaults to False.
            kekulize (bool): kekulizes SMILES (implicit aromaticity only).
                Defaults to False.
            all_bonds_explicit (bool): Makes all bonds explicit. Defaults to
                False, only applies if kekulize = True.
            all_hs_explicit (bool): Makes all hydrogens explicit. Defaults to
                False, only applies if kekulize = True.
            randomize (bool): perform a true randomization of SMILES tokens.
                Defaults to False.
            remove_bonddir (bool): Remove directional info of bonds.
                Defaults to False.
            remove_chirality (bool): Remove chirality information.
                Defaults to False.
            selfies (bool): Whether selfies is used instead of smiles, defaults
                to False.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
            chunk_size (int): size of the chunks. Defauls to 10000.
        """
        self.chunk_size = chunk_size
        super(_SMILESLazyDataset, self).__init__(
            smi_filepaths=smi_filepaths,
            smiles_language=smiles_language,
            padding=padding,
            padding_length=padding_length,
            add_start_and_stop=add_start_and_stop,
            canonical=canonical,
            augment=augment,
            kekulize=kekulize,
            all_bonds_explicit=all_bonds_explicit,
            all_hs_explicit=all_hs_explicit,
            randomize=randomize,
            remove_bonddir=remove_bonddir,
            remove_chirality=remove_chirality,
            selfies=selfies,
            device=device
        )

    def _setup_dataset(self) -> None:
        """Setup the dataset."""
        self._dataset = concatenate_file_based_datasets(
            filepaths=self.smi_filepaths,
            dataset_class=_SmiLazyDataset,
            chunk_size=self.chunk_size
        )
