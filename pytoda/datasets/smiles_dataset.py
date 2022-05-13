"""Datasets for smiles and transformations of smiles."""
import logging

import torch

from pytoda.warnings import device_warning

from ..smiles.processing import split_selfies
from ..smiles.smiles_language import SMILESLanguage, SMILESTokenizer
from ._smi_eager_dataset import _SmiEagerDataset
from ._smi_lazy_dataset import _SmiLazyDataset
from .base_dataset import DatasetDelegator
from .utils import concatenate_file_based_datasets

logger = logging.getLogger(__name__)


SMILES_DATASET_IMPLEMENTATIONS = {  # get class and acceptable keywords
    'eager': (_SmiEagerDataset, {'index_col', 'names'}),
    'lazy': (_SmiLazyDataset, {'chunk_size', 'index_col', 'names'}),
}  # name cannot be passed


class SMILESDataset(DatasetDelegator):
    """Dataset of SMILES."""

    def __init__(
        self,
        *smi_filepaths: str,
        backend: str = 'eager',
        name: str = 'smiles-dataset',
        device: torch.device = None,
        **kwargs,
    ) -> None:
        """
        Initialize a SMILES dataset.

        Args:
            smi_filepaths (Files): paths to .smi files.
            name (str): name of the SMILESDataset.
            backend (str): memory management backend.
                Defaults to eager, prefer speed over memory consumption.
            device (torch.device): DEPRECATED
            kwargs (dict): additional arguments for dataset constructor.
        """
        device_warning(device)
        # Parse language object and data paths
        self.smi_filepaths = smi_filepaths
        self.backend = backend
        self.name = name

        dataset_class, valid_keys = SMILES_DATASET_IMPLEMENTATIONS[self.backend]
        self.kwargs = dict((k, v) for k, v in kwargs.items() if k in valid_keys)
        self.kwargs['name'] = 'SMILES'

        self.dataset = concatenate_file_based_datasets(
            filepaths=self.smi_filepaths, dataset_class=dataset_class, **self.kwargs
        )

        DatasetDelegator.__init__(self)  # delegate to self.dataset
        if self.has_duplicate_keys:
            raise KeyError('Please remove duplicates from your .smi file.')


class SMILESTokenizerDataset(DatasetDelegator):
    """Dataset of token indexes from SMILES."""

    def __init__(
        self,
        *smi_filepaths: str,
        smiles_language: SMILESLanguage = None,
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
        vocab_file: str = None,
        iterate_dataset: bool = True,
        backend: str = 'eager',
        device: torch.device = None,
        name: str = 'smiles-encoder-dataset',
        **kwargs,
    ) -> None:
        """
        Initialize a dataset providing token indexes from source SMILES.

        The datasets transformations on smiles and encodings can be adapted,
        depending on the smiles_language used (see SMILESTokenizer).

        Args:
            smi_filepaths (Files): paths to .smi files.
            smiles_language (SMILESLanguage): a smiles language that transforms
                and encodes SMILES to token indexes. Defaults to None, where
                a SMILESTokenizer is instantited with the following arguments.
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
            sanitize (bool): RDKit sanitization of the molecule.
                Defaults to True.
            add_start_and_stop (bool): add start and stop token indexes.
                Defaults to False.
            padding (bool): pad sequences to longest in the smiles language.
                Defaults to True.
            padding_length (int): padding to match manually set length,
                applies only if padding is True. Defaults to None.
            vocab_file (str): Optional .json to load vocabulary. Defaults to
                None.
            iterate_dataset (bool): whether to go through all SMILES in the
                dataset to extend/build vocab, find longest sequence, and
                checks the passed padding length if applicable. Defaults to
                True.
            backend (str): memory management backend.
                Defaults to eager, prefer speed over memory consumption.
            name (str): name of the SMILESTokenizerDataset.
            device (torch.device): DEPRECATED
            kwargs (dict): additional arguments for dataset constructor.

        """
        device_warning(device)
        self.name = name
        self.dataset = SMILESDataset(*smi_filepaths, backend=backend, **kwargs)
        DatasetDelegator.__init__(self)  # delegate to self.dataset

        if smiles_language is not None:
            self.smiles_language = smiles_language
            params = (
                "canonical, augment, kekulize, all_bonds_explicit, selfies, sanitize, "
                "all_hs_explicit, remove_bonddir, remove_chirality, randomize, "
                "add_start_and_stop, padding, padding_length"
            )
            logger.error(
                'Since you provided a smiles_language, the following parameters to this'
                f' class will be ignored: {params}.\nHere are the problems:'
            )
            mismatch = False
            for p in params.split(','):
                if eval(p.strip()) != eval(f'smiles_language.{p.strip()}'):
                    logger.error(
                        f'Provided arg {p.strip()}:{eval(p.strip())} does not match the '
                        f'smiles_language value: {eval(f"smiles_language.{p.strip()}")}'
                        ' NOTE: smiles_language value takes preference!!'
                    )
                    mismatch = True
            if not mismatch:
                logger.error('Looking great, no problems found!')
            else:
                logger.error(
                    'To get rid of this, adapt the smiles_language *offline*, feed it'
                    'ready for intended usage, and adapt the constructor args to be '
                    'identical with their equivalents in the language object'
                )

        else:
            language_kwargs = {}  # SMILES default
            if selfies:
                language_kwargs = dict(
                    name='selfies-language', smiles_tokenizer=split_selfies
                )
            self.smiles_language = SMILESTokenizer(
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
            )

        if vocab_file:
            self.smiles_language.load_vocabulary(vocab_file)

        if iterate_dataset:
            # uses the smiles transforms
            self.smiles_language.add_dataset(self.dataset)

        try:
            if (
                self.smiles_language.padding
                and self.smiles_language.padding_length is None
            ):
                try:
                    # max_sequence_token_length has to be set somehow
                    if smiles_language is not None or iterate_dataset:
                        self.smiles_language.set_max_padding()
                except AttributeError:
                    raise TypeError(
                        'Setting a maximum padding length requires a '
                        'smiles_language with `set_max_padding` method. See '
                        '`SMILESTokenizer`.'
                    )
        except AttributeError:
            # SmilesLanguage w/o padding support passed.
            pass

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            torch.Tensor: a torch tensor of token indexes,
                for the current sample.
        """
        return self.smiles_language.smiles_to_token_indexes(self.dataset[index])
