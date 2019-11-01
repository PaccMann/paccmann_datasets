"""Implementation of _SMILESDataset."""
import torch
from torch.utils.data import Dataset
from ..types import FileList
from ..smiles.smiles_language import SMILESLanguage
from ..smiles.transforms import (
    SMILESToTokenIndexes, LeftPadding, ToTensor, SMILESRandomization
)
from ..transforms import Compose


class _SMILESDataset(Dataset):
    """
    SMILES dataset abstract definition.

    The implementation is abstract and can be extend to define different
    data loading policies.
    """

    def __init__(
        self,
        smi_filepaths: FileList,
        smiles_language: SMILESLanguage = None,
        padding: bool = True,
        padding_length: int = None,
        add_start_and_stop: bool = False,
        augment: bool = False,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu')
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
            augment (bool): perform SMILES augmentation. Defaults to False.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
        """
        Dataset.__init__(self)
        self.smi_filepaths = smi_filepaths
        if smiles_language is None:
            self.smiles_language = SMILESLanguage(
                add_start_and_stop=add_start_and_stop
            )
            self.smiles_language.add_smis(self.smi_filepaths)
        else:
            self.smiles_language = smiles_language
        self.padding = padding
        self.augment = augment
        self.padding_length = self.padding_length = (
            self.smiles_language.max_token_sequence_length
            if padding_length is None else padding_length
        )
        self.device = device
        transforms = [
            SMILESToTokenIndexes(smiles_language=self.smiles_language)
        ]
        if self.augment:
            transforms = [SMILESRandomization()] + transforms
        if self.padding:
            transforms.append(
                LeftPadding(
                    padding_length=self.padding_length,
                    padding_index=self.smiles_language.padding_index
                )
            )
        transforms.append(ToTensor(device=self.device))
        self.transform = Compose(transforms)
        self._dataset = None
        # NOTE: the dataset will be initialized and
        # designed to return SMILES strings
        self._setup_dataset()
        # NOTE: recover sample and index mappings
        self.sample_to_index_mapping = {}
        self.index_to_sample_mapping = {}
        for index in range(len(self._dataset)):
            dataset_index, sample_index = self._dataset.get_index_pair(index)
            dataset = self._dataset.datasets[dataset_index]
            sample = dataset.index_to_sample_mapping[sample_index]
            self.sample_to_index_mapping[sample] = index
            self.index_to_sample_mapping[index] = sample

    def _setup_dataset(self) -> None:
        """Setup the dataset."""
        raise NotImplementedError

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
