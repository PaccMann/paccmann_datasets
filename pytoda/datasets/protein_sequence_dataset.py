"""Implementation of ProteinSequenceDataset."""
import logging
from typing import Dict, Iterable

import numpy as np
import torch
from importlib_resources import files

from pytoda.warnings import device_warning

from ..proteins.protein_feature_language import ProteinFeatureLanguage
from ..proteins.protein_language import ProteinLanguage
from ..proteins.transforms import (
    MutateResidues,
    ProteinAugmentFlipSubstrs,
    ProteinAugmentSwapSubstrs,
    ReplaceByFullProteinSequence,
    SequenceToTokenIndexes,
)
from ..transforms import (
    AugmentByReversing,
    Compose,
    DiscardLowercase,
    ExtractFromDict,
    LeftPadding,
    ListToTensor,
    Randomize,
    ToTensor,
    ToUpperCase,
)
from ._fasta_eager_dataset import _FastaEagerDataset
from ._fasta_lazy_dataset import _FastaLazyDataset
from ._smi_eager_dataset import _SmiEagerDataset
from ._smi_lazy_dataset import _SmiLazyDataset
from .base_dataset import DatasetDelegator, KeyDataset
from .utils import concatenate_file_based_datasets

logger = logging.getLogger(__name__)

SEQUENCE_DATASET_IMPLEMENTATIONS = {  # get class and acceptable keywords
    '.csv': {
        'eager': (_SmiEagerDataset, {'index_col', 'names'}),
        'lazy': (_SmiLazyDataset, {'chunk_size', 'index_col', 'names'}),
    },  # .smi like .csv
    '.smi': {
        'eager': (_SmiEagerDataset, {'index_col', 'names'}),
        'lazy': (_SmiLazyDataset, {'chunk_size', 'index_col', 'names'}),
    },
    '.fasta': {
        'eager': (_FastaEagerDataset, {'gzipped', 'name'}),
        'lazy': (
            _FastaLazyDataset,
            {
                'name',
                # args to pyfaidx.Fasta
                'default_seq',
                'key_function',
                'as_raw',
                'strict_bounds',
                'read_ahead',
                'mutable',
                'split_char',
                'duplicate_action',
                'filt_function',
                'one_based_attributes',
                'read_long_names',
                'sequence_always_upper',
                'rebuild',
                'build_index',
            },
        ),
    },
    '.fasta.gz': {
        'eager': (_FastaEagerDataset, {'gzipped', 'name'}),
        # requires Biopython installation
        'lazy': (
            _FastaLazyDataset,
            {
                'name',
                # args to pyfaidx.Fasta
                'default_seq',
                'key_function',
                'as_raw',
                'strict_bounds',
                'read_ahead',
                'mutable',
                'split_char',
                'duplicate_action',
                'filt_function',
                'one_based_attributes',
                'read_long_names',
                'sequence_always_upper',
                'rebuild',
                'build_index',
            },
        ),
    },
}


def protein_sequence_dataset(
    *filepaths: str, filetype: str, backend: str, **kwargs
) -> KeyDataset:
    """
    Return a dataset of protein sequences.

    Args:
        filepaths (Iterable[str]): Paths to the files containing the protein sequences.
        filetype (str): The filetype of the protein sequences.
        backend (str): The backend to use for the dataset.
        kwargs: Keyword arguments for the dataset.
    """
    try:
        # hardcoded factory
        dataset_class, valid_keys = SEQUENCE_DATASET_IMPLEMENTATIONS[filetype][backend]
    except KeyError:
        raise ValueError(  # filetype checked already
            f'backend {backend} not supported for {filetype}.'
        )

    kwargs['gzipped'] = True if filetype == '.fasta.gz' else False
    # prune unsupported arguments
    kwargs = dict((k, v) for k, v in kwargs.items() if k in valid_keys)
    kwargs['name'] = 'Sequence'

    return concatenate_file_based_datasets(
        filepaths=filepaths, dataset_class=dataset_class, **kwargs
    )


class ProteinSequenceDataset(DatasetDelegator):
    """Protein Sequence dataset using a Language to transform sequences."""

    def __init__(
        self,
        *filepaths: str,
        filetype: str = '.smi',
        protein_language: ProteinLanguage = None,
        amino_acid_dict: str = 'iupac',
        padding: bool = True,
        padding_length: int = None,
        add_start_and_stop: bool = False,
        augment_by_revert: bool = False,
        sequence_augment: Dict = {},
        protein_keep_only_uppercase: bool = False,
        randomize: bool = False,
        backend: str = 'eager',
        iterate_dataset: bool = False,
        name: str = 'protein-sequences',
        device: torch.device = None,
        **kwargs,
    ) -> None:
        """
        Initialize a Protein Sequence dataset.

        Args:
            *filepaths (Files): paths to .smi, .csv/.fasta/.fasta.gz file
                with the sequences.
            filetype (str): From {.smi, .csv, .fasta, .fasta.gz}.
            protein_language (ProteinLanguage): a ProteinLanguage (or child)
                instance, e.g. ProteinFeatureLanguage. Defaults to None,
                creating a default instance.
            amino_acid_dict (str): Type of dictionary used for amino acid sequences.
                Defaults to 'iupac', alternative is 'unirep' or 'human-kinase-alignment'
            padding (bool): pad sequences to longest in the protein language.
                Defaults to True.
            padding_length (int): manually sets number of applied paddings,
                applies only if padding is True. Defaults to None.
            add_start_and_stop (bool): add start and stop token indexes.
                Defaults to False.
            augment_by_revert (bool): perform Protein augmentation by reverting
                Sequences. Defaults to False.
            randomize (bool): perform a true randomization of Protein tokens.
                Defaults to False.
            sequence_augment (Dict): a dictionary to specify additional sequence
                augmentation. Defaults to {}. Items can be:
                    `alignment_path`: A path (str) to a `.smi` (or `.tsv`) file
                        with alignment information that specifies which residues of the
                        sequence are to be used. E.g., to extract active site sequences
                        from full proteins. Do not use a header in the file.
                        1. column has to be the full protein sequence (use upper
                            case only for residues to be used). E.g., ggABCggDEFgg
                        2. column has to be the condensed sequence (ABCDEF).
                        3. column has to be the protein identifier.
                        NOTE: Such a file is *necessary* to apply all augmentation types
                        specified in this dictionary (`sequence_augment`).
                        NOTE: Unless specified, this defaults to
                        `kinase_activesite_alignment.smi`, a file in
                        `pytoda.proteins.metadata` that can ONLY be used to
                        extract active site sequences of human kinases (based on the
                        active site definition in Sheridan et al. (2009, JCIM) and
                        Martin & Mukherjee (2012, JCIM).
                    `discard_lowercase`: A (bool) specifying whether all lowercase
                        characters (residues) in the sequence should be discarded.
                        NOTE: This defaults to True.
                    `flip_substrings`: A probability (float) to flip each contiguous
                        upper-case substring (e.g., an active site substring that lies
                        contiguously in the original sequence).
                        Defaults to 0.0, i.e., no flipping.
                        E.g., ABCDEF could become CBADEF or CBAFED or ABCFED if the
                        original sequence is ggABCggDEFgg.
                    `swap_substrings`: A probability (float) to swap neighboring
                        substrings. Defaults to 0.0, i.e., no swapping.
                        E.g., ABCDEF could become DEFABC if the original sequence is
                        ggABCggDEFgg.
                    `noise`: A 2-Tuple of (float, float) that specifies the probability
                        for a random, single-residue mutation inside and outside the
                        relevent part. Defaults to (0.0, 0.0), i.e., no noise.
                        E.g., with (0.0, 0.5),  ggABCggDEFgg could become hgABCgbDEFgg.
            iterate_dataset (bool): whether to go through all items in the dataset
                to detect unknown characters, find longest sequence and checks
                passed padding length if applicable. Defaults to False.
            backend (str): memory management backend.
                Defaults to eager, prefer speed over memory consumption.
            name (str): name of the ProteinSequenceDataset.
            device (torch.device): DEPRECATED
            kwargs (dict): additional arguments for dataset constructor.
        """

        device_warning(device)
        # Parse language object and data paths
        self.filepaths = filepaths
        self.filetype = filetype
        self.backend = backend

        assert filetype in [
            '.csv',
            '.smi',
            '.fasta',
            '.fasta.gz',
        ], f'Unknown filetype given {filetype}'
        self.name = name

        # setup dataset
        self._setup_dataset(**kwargs)
        self.cumulative_sizes = np.cumsum([0] + [len(s) for s in self.datasets])

        DatasetDelegator.__init__(self)  # delegate to self.dataset
        if self.has_duplicate_keys:
            raise KeyError(f'Please remove duplicates from your {self.filetype} file.')

        if protein_language is None:
            self.protein_language = ProteinLanguage(
                amino_acid_dict=amino_acid_dict, add_start_and_stop=add_start_and_stop
            )
        else:
            self.protein_language = protein_language
            assert (
                add_start_and_stop == protein_language.add_start_and_stop
            ), f'add_start_and_stop was "{add_start_and_stop}", but given '
            f'Protein Language has {protein_language.add_start_and_stop}.'

        if iterate_dataset or not protein_language:
            tokens = set(self.protein_language.token_to_index.keys())
            for sequence in self.dataset:
                # sets max_token_sequence_length
                self.protein_language.add_sequence(sequence)
                seq_tokens = set(list(sequence))
                if seq_tokens - tokens != set():
                    logger.error(
                        'Found unknown token(s): %s', list(seq_tokens - tokens)
                    )
        elif (
            not iterate_dataset
            and protein_language
            and protein_language.max_token_sequence_length < 3
        ):
            raise ValueError(
                'If provided ProteinLanguage is empty, set iterate_dataset to True'
            )

        # Set up transformation paramater
        self.padding = padding
        self.padding_length = (
            self.protein_language.max_token_sequence_length
            if padding_length is None
            else padding_length
        )
        if self.padding_length < self.protein_language.max_token_sequence_length:
            logger.warning(
                f'WARNING: Passed padding length was {padding_length} but '
                'protein language has padding length: '
                f'{self.protein_language.max_token_sequence_length}. '
                'Some sequences might get truncated.'
            )
        self.randomize = randomize
        self.augment_by_revert = augment_by_revert

        # Sequence-based augmentation
        self.sequence_augment = sequence_augment
        transforms = self.setup_sequence_augmentation(sequence_augment)

        if self.augment_by_revert:
            transforms += [AugmentByReversing()]

        self.language_transforms = Compose(transforms.copy())

        transforms += [SequenceToTokenIndexes(protein_language=self.protein_language)]
        if self.randomize:
            transforms += [Randomize()]
        if self.padding:
            transforms += [
                LeftPadding(
                    padding_length=self.padding_length,
                    padding_index=self.protein_language.token_to_index['<PAD>'],
                )
            ]
        if isinstance(self.protein_language, ProteinFeatureLanguage):
            transforms += [ListToTensor()]
        elif isinstance(self.protein_language, ProteinLanguage):
            transforms += [ToTensor()]
        else:
            raise TypeError(
                'Please choose either ProteinLanguage or ' 'ProteinFeatureLanguage'
            )
        self.transform = Compose(transforms)

    def _setup_dataset(self, **kwargs) -> None:
        """Setup the dataset."""
        self.dataset = protein_sequence_dataset(
            *self.filepaths, filetype=self.filetype, backend=self.backend, **kwargs
        )

    def setup_sequence_augmentation(self, sequence_augment: Dict):
        """
        Setup the sequence augmentation.

        Args:
            sequence_augment: A dictionary to specify the sequence augmentation
                strategies. For details see the constructor docs.
        """
        if not isinstance(sequence_augment, Dict):
            raise TypeError(
                f"Pass sequence_augment as Dict not {type(sequence_augment)}"
            )

        # Build up cascade of Protein transformations
        transforms = []

        # No work to do
        if sequence_augment == {}:
            transforms += [ExtractFromDict(key='sequence')]
            return transforms

        # Start composing transforms
        self.alignment_path = sequence_augment.pop(
            'alignment_path',
            files('pytoda.proteins.metadata').joinpath(
                'kinase_activesite_alignment.smi'
            ),
        )

        transforms += [ReplaceByFullProteinSequence(self.alignment_path)]

        # Flip substrings
        self.flip_substrings = sequence_augment.pop('flip_substrings', 0.0)
        transforms += [ProteinAugmentFlipSubstrs(p=self.flip_substrings)]

        # Swap substrings
        self.swap_substrings = sequence_augment.pop('swap_substrings', 0.0)
        transforms += [ProteinAugmentSwapSubstrs(self.swap_substrings)]

        # Inject noise (single residue mutation)
        self.noise = sequence_augment.pop('noise', (0.0, 0.0))
        if not isinstance(self.noise, Iterable) or len(self.noise) != 2:
            raise TypeError(f'Noise has to be Iterable of length 2 not {self.noise}')
        transforms += [MutateResidues(*self.noise)]

        # Prune lowercase characters
        self.discard_lowercase = sequence_augment.pop('discard_lowercase', True)
        if self.discard_lowercase:
            transforms += [DiscardLowercase()]

        """
        In the future this could be removed to distinguish between upper/lowercase
        characters but it would necessitate changes in protein language.
        """
        # Always convert to uppercase
        transforms += [ToUpperCase()]

        for k, v in sequence_augment.items():
            logger.warning(f"Ignoring sequence_augment item: {k}: {v}.")

        return transforms

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            torch.Tensor: a torch tensor of token indexes,
                for the current sample.
        """
        sample_dict = dict(sequence=self.dataset[index], id=self.dataset.get_key(index))
        return self.transform(sample_dict)
