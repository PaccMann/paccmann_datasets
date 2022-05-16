"""Implementation of ProteinSequenceDataset."""
import logging

import torch

from pytoda.warnings import device_warning
from typing import List, Optional
from ..proteins.protein_feature_language import ProteinFeatureLanguage
from ..proteins.protein_language import ProteinLanguage
from ..proteins.transforms import SequenceToTokenIndexes
from ..transforms import (
    AugmentByReversing,
    Compose,
    LeftPadding,
    ListToTensor,
    Randomize,
    ToTensor,
)
from ..proteins.transforms import (
    LoadActiveSiteAlignmentInfo,
    ProteinAugmentFlipActiveSiteSubstrs,
    ProteinAugmentActiveSiteGuidedNoise,
    ProteinAugmentSwitchBetweenActiveSiteSubstrs,
    KeepOnlyUpperCase,
    ToUpperCase,
    ExtractFromDict,
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
    """Return a dataset of protein sequences."""
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
    """
    Protein Sequence dataset using a Language to transform sequences.

    """

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
        load_active_site_alignment_info: Optional[List[str]] = None,
        protein_augment_flip_active_site_substrs: Optional[float] = None,
        protein_augment_active_site_guided_noise: Optional[List[float]] = None,
        protein_augment_switch_between_active_site_substrs: Optional[float] = None,
        protein_keep_only_uppercase:bool = False,        
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
            load_active_site_alignment_info (Optional[str]): load active site alignment info (for example - converts and active site like LGKGTFGKVAKELLTLFVMEYVNGGEFIVENMTDL into fdylklLGKGTFGKVilvrekasgkyyAmKilkkeviiakdeva...)
            protein_augment_flip_active_site_substrs Optional[float]: randomly flips with the given probability each consecutive active site substring
            protein_augment_active_site_guided_noise Optional[List[float]]: injects (optionally different) noise into residues inside and outside the active site.
                expects the list to contain two float values - [MUTATION_PROBABILITY_INSIDE_ACTIVE_SITE, MUTATION_PROBABILITY_OUTSIDE_ACTIVE_SITE]
            protein_augment_switch_between_active_site_substrs Optional[float]: randomly switches places between neighbour active site substrings
            protein_keep_only_uppercase (bool): default=False, keep only uppercase letters and discard all the rest
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

        # Build up cascade of Protein transformations
        transforms = []        
        
        ##### related to active site guided augmentation
        self.load_active_site_alignment_info = load_active_site_alignment_info
        self.protein_augment_flip_active_site_substrs = protein_augment_flip_active_site_substrs
        self.protein_augment_active_site_guided_noise = protein_augment_active_site_guided_noise
        self.protein_augment_switch_between_active_site_substrs = protein_augment_switch_between_active_site_substrs
        self.protein_keep_only_uppercase = protein_keep_only_uppercase

        if self.load_active_site_alignment_info is not None:
            assert isinstance(self.load_active_site_alignment_info, str)
            transforms += [LoadActiveSiteAlignmentInfo(self.load_active_site_alignment_info)]
        else:
            transforms += [ExtractFromDict(key='sequence')]

        if self.protein_augment_flip_active_site_substrs is not None:
            assert self.load_active_site_alignment_info is not None
            assert isinstance(self.protein_augment_flip_active_site_substrs, float)
            transforms += [ProteinAugmentFlipActiveSiteSubstrs(p=self.protein_augment_flip_active_site_substrs)]

        if self.protein_augment_active_site_guided_noise is not None:
            assert self.load_active_site_alignment_info is not None
            assert isinstance(self.protein_augment_active_site_guided_noise, list)
            assert 2 == len(protein_augment_active_site_guided_noise)
            transforms += [ProteinAugmentActiveSiteGuidedNoise(*self.protein_augment_active_site_guided_noise)]

        if self.protein_augment_switch_between_active_site_substrs is not None:
            assert self.load_active_site_alignment_info is not None
            assert isinstance(self.protein_augment_switch_between_active_site_substrs, float)
            transforms += [ProteinAugmentSwitchBetweenActiveSiteSubstrs(self.protein_augment_switch_between_active_site_substrs)]
            
        if self.protein_keep_only_uppercase:
            transforms += [KeepOnlyUpperCase()]
            
        #TODO: this can be theoretically done always
        if self.load_active_site_alignment_info is not None:
            transforms += [ToUpperCase()]

        #note - it's important to keep "augment_by_revert" after this section and not before it.
        ##########

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

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            torch.Tensor: a torch tensor of token indexes,
                for the current sample.
        """      
        #import ipdb;ipdb.set_trace()  
        #since multiple identical active sites have different full sequence, passing protein id as well

        assert 1==len(self.dataset.datasets)
        extracted = self.dataset.datasets[0].df.iloc[index]
        assert extracted.Sequence == self.dataset[index]        

        sample_dict = dict(sequence=extracted.Sequence, protein_id=extracted.name)
        return self.transform(sample_dict)
