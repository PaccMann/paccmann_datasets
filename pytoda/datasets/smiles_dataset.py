"""Datasets for smiles and transformations of smiles."""
import logging

import torch

from ..smiles.processing import tokenize_selfies
from ..smiles.smiles_language import SMILESEncoder
from ..types import FileList
from ._smi_eager_dataset import _SmiEagerDataset
from ._smi_lazy_dataset import _SmiLazyDataset
from .base_dataset import DatasetDelegator
from .utils import concatenate_file_based_datasets

logger = logging.getLogger(__name__)


SMILES_DATASET_IMPLEMENTATIONS = {  # get class and acceptable keywords
    'eager': (_SmiEagerDataset, {'name'}),
    'lazy': (_SmiLazyDataset, {'chunk_size', 'name'}),
}


class SMILESDataset(DatasetDelegator):
    """ Dataset of SMILES. """

    def __init__(
        self,
        *smi_filepaths: FileList,
        backend: str = 'eager',
        name: str = 'smiles-dataset',
        **kwargs
    ) -> None:
        """
        Initialize a SMILES dataset.

        Args:
            smi_filepaths (FileList): paths to .smi files.
            name (str): name of the SMILESDataset.
            backend (str): memory management backend.
                Defaults to eager, prefer speed over memory consumption.
            kwargs (dict): additional arguments for dataset constructor.
        """
        # Parse language object and data paths
        self.smi_filepaths = smi_filepaths
        self.backend = backend
        self.name = name

        dataset_class, valid_keys = SMILES_DATASET_IMPLEMENTATIONS[
            self.backend
        ]
        kwargs = dict(
            (k, v) for k, v in kwargs.items() if k in valid_keys
        )

        self.dataset = concatenate_file_based_datasets(
            filepaths=self.smi_filepaths,
            dataset_class=dataset_class,
            **kwargs
        )

        DatasetDelegator.__init__(self)  # delegate to self.dataset
        if self.has_duplicate_keys:
            raise KeyError('Please remove duplicates from your .smi file.')


class SMILESEncoderDataset(DatasetDelegator):
    """ Dataset of token indices from SMILES. """

    def __init__(
        self,
        *smi_filepaths: FileList,
        smiles_language: SMILESEncoder = None,
        canonical: bool = False,
        augment: bool = False,
        kekulize: bool = False,
        all_bonds_explicit: bool = False,
        all_hs_explicit: bool = False,
        remove_bonddir: bool = False,
        remove_chirality: bool = False,
        selfies: bool = False,
        sanitize: bool = True,
        randomize: bool = False,
        add_start_and_stop: bool = False,
        padding: bool = True,
        padding_length: int = None,
        iterate_dataset: bool = True,
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
        backend: str = 'eager',
        name: str = 'smiles-encoder-dataset',
        **kwargs
    ) -> None:
        """
        Initialize a dataset providing token indices from source SMILES.

        The datasets transformations on smiles and encodings can be adapted,
        depending on the smiles_language used (see SMILESEncoder).

        Args:
            smi_filepaths (FileList): paths to .smi files.
            smiles_language (SMILESEncoder): a smiles language that transforms
                and encodes SMILES to token indices. Defaults to None, where
                a SMILESEncoder is instantited with the following arguments.
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
            add_start_and_stop (bool): add start and stop token indexes.
                Defaults to False.
            padding (bool): pad sequences to longest in the smiles language.
                Defaults to True.
            padding_length (int): padding to match manually set length,
                applies only if padding is True. Defaults to None.
            iterate_dataset (bool): whether to go through all SMILES in the
                dataset to build/extend vocab, find longest sequence, and
                checks the passed padding length if applicable. Defaults to
                True.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
            backend (str): memory management backend.
                Defaults to eager, prefer speed over memory consumption.
            name (str): name of the SMILESEncoderDataset.
            kwargs (dict): additional arguments for dataset constructor.

        """
        self.name = name
        self.dataset = SMILESDataset(
            *smi_filepaths,
            backend=backend,
            **kwargs
        )
        DatasetDelegator.__init__(self)  # delegate to self.dataset

        if smiles_language is not None:
            self.smiles_language = smiles_language
        else:
            language_kwargs = {}  # SMILES default
            if selfies:
                language_kwargs = dict(
                    name='selfies-language',
                    smiles_tokenizer=lambda selfies: tokenize_selfies(selfies)
                )
            self.smiles_language = SMILESEncoder(
                **language_kwargs,
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
                add_start_and_stop=add_start_and_stop,
                padding=padding,
                padding_length=padding_length,
                device=device,
            )
        if iterate_dataset:  # TODO why not iterate smis for selfies before?
            # uses the smiles transforms
            self.smiles_language.add_dataset(self.dataset)
            self.smiles_language.set_smiles_transforms()  # TODO want/need?
            self.smiles_language.add_dataset(self.dataset)
            if padding:
                checked_padding = self.smiles_language.checked_padding_length()
                self.smiles_language.padding_length = checked_padding
                if checked_padding != padding_length:
                    logger.warning(
                        'From passing over the dataset the given padding '
                        f'length was set to {checked_padding} to not trunkate '
                        'any sequences.'
                    )

        self.smiles_language.set_initial_transforms()

    def __getitem__(self, index: int) -> torch.tensor:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            torch.tensor: a torch tensor of token indexes,
                for the current sample.
        """
        return self.smiles_language.smiles_to_token_indexes(
            self.dataset[index]
        )
