"""Implementation of ProteinSequenceDataset."""
import torch
from torch.utils.data import Dataset

from ..proteins.protein_language import ProteinLanguage
from ..proteins.transforms import AugmentByReversing, SequenceToTokenIndexes
from ..smiles.transforms import LeftPadding, Randomize, ToTensor
from ..transforms import Compose
from ..types import FileList
from ._smi_eager_dataset import _SmiEagerDataset
from .utils import concatenate_file_based_datasets


class ProteinSequenceDataset(Dataset):
    """
    Protein Sequence dataset definition.

    """

    def __init__(
        self,
        *filepaths: FileList,
        protein_language: ProteinLanguage = None,
        amino_acid_dict: str = 'iupac',
        padding: bool = True,
        padding_length: int = None,
        add_start_and_stop: bool = False,
        augment_by_revert: bool = False,
        randomize: bool = False,
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
    ) -> None:
        """
        Initialize a Protein Sequence dataset.

        Args:
            filepaths (FileList): paths to .smi/.csv file with the sequences.
            protein_language (ProteinLanguage): a protein language or child
                object. Defaults to None.
            amino_acid_dict (str): Type of dictionary used for amino acid
                sequences. Defaults to 'iupac', alternative is 'unirep'.
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
        """
        Dataset.__init__(self)
        # Parse language object and data paths
        self.filepaths = filepaths

        if protein_language is None:
            self.protein_language = ProteinLanguage(
                amino_acid_dict=amino_acid_dict,
                add_start_and_stop=add_start_and_stop
            )
        else:
            self.protein_language = protein_language
            assert (
                add_start_and_stop == protein_language.add_start_and_stop
            ), f'add_start_and_stop was "{add_start_and_stop}", but given '
            f'Protein Language has {protein_language.add_start_and_stop}.'

        # Set up transformation paramater
        self.padding = padding
        self.padding_length = self.padding_length = (
            self.protein_language.max_token_sequence_length
            if padding_length is None else padding_length
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
        self._setup_dataset()
        # Run once over dataset to add missing tokens to smiles language
        for index in range(len(self._dataset)):
            self.protein_language.add_sequence(
                self.language_transforms(self._dataset[index])
            )
        transforms = _transforms.copy()
        transforms += [
            SequenceToTokenIndexes(protein_language=self.protein_language)
        ]
        if self.randomize:
            transforms += [Randomize()]
        if self.padding:
            if padding_length is None:
                self.padding_length = (
                    self.protein_language.max_token_sequence_length
                )
            transforms += [
                LeftPadding(
                    padding_length=self.padding_length,
                    padding_index=self.protein_language.token_to_index['<PAD>']
                )
            ]
        transforms += [ToTensor(device=self.device)]
        self.transform = Compose(transforms)

        # NOTE: recover sample and index mappings
        self.sample_to_index_mapping = {}
        self.index_to_sample_mapping = {}

        for index in range(len(self._dataset)):
            dataset_index, sample_index = self._dataset.get_index_pair(index)
            dataset = self._dataset.datasets[dataset_index]
            try:
                sample = dataset.index_to_sample_mapping[sample_index]
            except KeyError:
                raise KeyError('Please remove duplicates from your .smi file.')
            self.sample_to_index_mapping[sample] = index
            self.index_to_sample_mapping[index] = sample

    def _setup_dataset(self) -> None:
        """Setup the dataset."""
        self._dataset = concatenate_file_based_datasets(
            filepaths=self.filepaths,
            dataset_class=_SmiEagerDataset,
            name='Sequence'
        )

    def __len__(self) -> int:
        """Total number of samples."""
        return len(self._dataset)

    def __getitem__(self, index: int) -> torch.tensor:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            torch.tensor: a torch tensor of token indexes,
                for the current sample.
        """
        return self.transform(self._dataset[index])
