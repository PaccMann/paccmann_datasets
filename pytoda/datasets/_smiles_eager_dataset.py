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
        augment: bool = False,
        kekulize: bool = False,
        allBondsExplicit: bool = False,
        allHsExplicit: bool = False,
        randomize: bool = False,
        remove_bonddir: bool = False,
        remove_chirality: bool = False,
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
            augment (bool): perform SMILES augmentation. Defaults to False.
            kekulize (bool): kekulizes SMILES (implicit aromaticity only).
                Defaults to False.
            allBondsExplicit (bool): Makes all bonds explicit. Defaults to
                False, only applies if kekulize = True.
            allHsExplicit (bool): Makes all hydrogens explicit. Defaults to
                False, only applies if kekulize = True.
            randomize (bool): perform a true randomization of SMILES tokens.
                Defaults to False.
            remove_bonddir (bool): Remove directional info of bonds.
                Defaults to False.
            remove_chirality (bool): Remove chirality information.
                Defaults to False.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
        """
        super(_SMILESEagerDataset, self).__init__(
            smi_filepaths=smi_filepaths,
            smiles_language=smiles_language,
            padding=padding,
            add_start_and_stop=add_start_and_stop,
            augment=augment,
            kekulize=kekulize,
            allBondsExplicit=allBondsExplicit,
            allHsExplicit=allHsExplicit,
            randomize=randomize,
            remove_bonddir=remove_bonddir,
            remove_chirality=remove_chirality,
            device=device
        )

    def _setup_dataset(self) -> None:
        """Setup the dataset."""
        self._dataset = concatenate_file_based_datasets(
            filepaths=self.smi_filepaths, dataset_class=_SmiEagerDataset
        )
        # Run once over dataset to add missing tokens to smiles language
        for index in range(len(self._dataset)):
            self.smiles_language.add_smiles(
                self.smiles_transforms(self._dataset[index])
            )
