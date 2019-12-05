"""Implementation of _SMILESEagerDataset."""
import torch
from ..types import FileList
from ..smiles.smiles_language import SMILESLanguage
from ._smiles_dataset import _SMILESDataset
from ._smi_eager_dataset import _SmiEagerDataset
from .utils import concatenate_file_based_datasets


class _SMILESEagerDataset(_SMILESDataset):
    """
    SMILES dataset using eager loading.

    Suggested when handling datasets that can fit in the device memory.
    In case of out of memory errors consider using _SMILESLazyDataset.
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
        device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> None:
        """
        Initialize a SMILES eager dataset.

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
        """
        super(_SMILESEagerDataset, self).__init__(
            smi_filepaths=smi_filepaths,
            smiles_language=smiles_language,
            padding_length=padding_length,
            padding=padding,
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
            filepaths=self.smi_filepaths, dataset_class=_SmiEagerDataset
        )
