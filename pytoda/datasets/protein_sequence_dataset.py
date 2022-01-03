"""Implementation of ProteinSequenceDataset."""
import logging

import torch

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
        randomize: bool = False,
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
        backend: str = 'eager',
        iterate_dataset: bool = False,
        name: str = 'protein-sequences',
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
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
            iterate_dataset (bool): whether to go through all items in the
                dataset to extend/build vocab, find longest sequence, and
                checks the passed padding length if applicable. Defaults to
                False.
            backend (str): memory management backend.
                Defaults to eager, prefer speed over memory consumption.
            name (str): name of the ProteinSequenceDataset.
            kwargs (dict): additional arguments for dataset constructor.
        """

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

        if iterate_dataset:
            tokens = set(self.protein_language.token_to_index.keys())
            for sequence in self.dataset:
                # sets max_token_sequence_length
                self.protein_language.add_sequence(sequence)
                seq_tokens = set(list(sequence))
                if seq_tokens - tokens != set():
                    logger.error(
                        'Found unknown token(s): %s', list(seq_tokens - tokens)
                    )

        # Set up transformation paramater
        self.padding = padding
        self.padding_length = self.padding_length = (
            self.protein_language.max_token_sequence_length
            if padding_length is None
            else padding_length
        )
        self.randomize = randomize
        self.augment_by_revert = augment_by_revert
        self.device = device

        # Build up cascade of Protein transformations
        # Below transformations are optional
        _transforms = []
        if self.augment_by_revert:
            _transforms += [AugmentByReversing()]
        self.language_transforms = Compose(_transforms)

        # Run once over dataset to add missing tokens to smiles language
        for index in range(len(self.dataset)):
            self.protein_language.add_sequence(
                self.language_transforms(self.dataset[index])
            )
        transforms = _transforms.copy()
        transforms += [SequenceToTokenIndexes(protein_language=self.protein_language)]
        if self.randomize:
            transforms += [Randomize()]
        if self.padding:
            if padding_length is None:
                self.padding_length = self.protein_language.max_token_sequence_length
            transforms += [
                LeftPadding(
                    padding_length=self.padding_length,
                    padding_index=self.protein_language.token_to_index['<PAD>'],
                )
            ]
        if isinstance(self.protein_language, ProteinFeatureLanguage):
            transforms += [ListToTensor(device=self.device)]
        elif isinstance(self.protein_language, ProteinLanguage):
            transforms += [ToTensor(device=self.device)]
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
        return self.transform(self.dataset[index])
