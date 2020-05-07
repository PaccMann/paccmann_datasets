"""SMILESDataset module."""
import torch
from torch.utils.data import Dataset
from ..smiles.smiles_language import SMILESLanguage
from ._smiles_eager_dataset import _SMILESEagerDataset
from ._smiles_lazy_dataset import _SMILESLazyDataset
from ..types import FileList
from .base_dataset import DatasetDelegator

SMILES_DATASET_IMPLEMENTATIONS = {
    'eager': _SMILESEagerDataset,
    'lazy': _SMILESLazyDataset
}


class SMILESDataset(DatasetDelegator):
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
        canonical: bool = False,
        augment: bool = False,
        kekulize: bool = False,
        all_bonds_explicit: bool = False,
        all_hs_explicit: bool = False,
        randomize: bool = False,
        remove_bonddir: bool = False,
        remove_chirality: bool = False,
        selfies: bool = False,
        sanitize: bool = True,
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
        backend: str = 'eager',
        name: str = 'smiles-dataset',
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
            canonical (bool): performs canonicalization of SMILES (one
                original string for one molecule), if True, then other
                transformations (augment etc, see below) do not apply
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
            sanitize (bool): Sanitize SMILES. Defaults to True.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
            backend (str): memeory management backend.
                Defaults to eager, prefer speed over memory consumption.
            name (str): name of the SMILESDataset.

        """
        DatasetDelegator.__init__(self)
        self.name = name
        if not (backend in SMILES_DATASET_IMPLEMENTATIONS):
            raise RuntimeError(
                'backend={} not supported! '.format(backend) +
                'Select one in [{}]'.
                format(','.join(SMILES_DATASET_IMPLEMENTATIONS.keys()))
            )
        self.dataset = SMILES_DATASET_IMPLEMENTATIONS[backend](
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
            sanitize=sanitize,
            device=device
        )
          # base_dataset: test for theses attributes:
        # self.smiles_language = self.dataset.smiles_language
        # self.sample_to_index_mapping = self.dataset.sample_to_index_mapping
