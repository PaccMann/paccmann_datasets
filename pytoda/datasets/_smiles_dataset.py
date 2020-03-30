"""Implementation of _SMILESDataset."""
import warnings

import torch
from rdkit import Chem
from torch.utils.data import Dataset

from ..smiles.processing import (
    SMILES_TOKENIZER, tokenize_selfies, tokenize_smiles
)
from ..smiles.smiles_language import SMILESLanguage
from ..smiles.transforms import (
    Augment, Canonicalization, Kekulize, LeftPadding, NotKekulize, Randomize,
    RemoveIsomery, Selfies, SMILESToTokenIndexes, ToTensor
)
from ..transforms import Compose
from ..types import FileList


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
        Initialize a SMILES dataset.

        Args:
            smi_filepaths (FileList): paths to .smi files.
            smiles_language (SMILESLanguage): a smiles language or child object
                Defaults to None.
            padding (bool): pad sequences to longest in the smiles language.
                Defaults to True.
            padding_length (int): manually sets number of applied paddings,
                applies only if padding is True. Defaults to None.
            add_start_and_stop (bool): add start and stop token indexes.
                Defaults to False.
            canonical (bool): performs canonicalization of SMILES (one original
                string for one molecule). If True, then other transformations
                (augment etc, see below) do not apply.
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

        NOTE: If the setup is too slow, consider skipping all SMILES language
            transforms. To achieve this, set ALL following arguments to False:
                - `canonical`
                - `augment`
                - `kekulize`
                - `all_bonds_explicit`
                - `all_hs_explicit`
                - `remove_bonddir`
                - `remove_chirality`
        """
        Dataset.__init__(self)
        # Parse language object and data paths
        self.smi_filepaths = smi_filepaths

        if smiles_language is None:
            self.smiles_language = SMILESLanguage(
                name='selfies-language' if selfies else 'smiles_language',
                smiles_tokenizer=(
                    (lambda selfies: tokenize_selfies(selfies)) if selfies else
                    (lambda smiles: tokenize_smiles(smiles, SMILES_TOKENIZER))
                ),
                add_start_and_stop=add_start_and_stop
            )
            if not selfies:
                self.smiles_language = SMILESLanguage(
                    add_start_and_stop=add_start_and_stop
                )
                self.smiles_language.add_smis(self.smi_filepaths)
        else:
            self.smiles_language = smiles_language

        # Set up transformation paramater
        self.padding = padding
        self.augment = augment
        self.padding_length = self.padding_length = (
            self.smiles_language.max_token_sequence_length
            if padding_length is None else padding_length
        )
        self.kekulize = kekulize
        self.canonical = canonical
        self.all_bonds_explicit = all_bonds_explicit
        self.all_hs_explicit = all_hs_explicit
        self.randomize = randomize
        self.remove_bonddir = remove_bonddir
        self.remove_chirality = remove_chirality
        self.selfies = selfies
        self.device = device

        # Build up cascade of SMILES transformations
        # Below transformations are optional
        language_transforms = []
        if self.canonical:
            language_transforms += [Canonicalization()]
        else:
            if self.remove_bonddir or self.remove_chirality:
                language_transforms += [
                    RemoveIsomery(
                        bonddir=self.remove_bonddir,
                        chirality=self.remove_chirality
                    )
                ]
            if self.kekulize:
                language_transforms += [
                    Kekulize(
                        all_bonds_explicit=self.all_bonds_explicit,
                        all_hs_explicit=self.all_hs_explicit
                    )
                ]
            elif self.all_bonds_explicit or self.all_hs_explicit:
                language_transforms += [
                    NotKekulize(
                        all_bonds_explicit=self.all_bonds_explicit,
                        all_hs_explicit=self.all_hs_explicit
                    )
                ]
            if self.augment:
                language_transforms += [
                    Augment(
                        kekule_smiles=self.kekulize,
                        all_bonds_explicit=self.all_bonds_explicit,
                        all_hs_explicit=self.all_hs_explicit
                    )
                ]
            if self.selfies:
                language_transforms += [Selfies()]

        self.language_transforms = Compose(language_transforms)
        num_tokens = len(self.smiles_language.token_to_index)
        self._setup_dataset()

        # If we use language transforms: add missing tokens to smiles language
        if (
            smiles_language is not None
            and len(self.language_transforms.transforms) == 0
        ):
            print(
                'WARNING: You operate in the fast-setup regime.\nIf you pass a'
                ' SMILESLanguage object, but dont specify any SMILES language'
                ' transform, no pass is given over all SMILES in '
                f'{self.smi_filepaths}.\nCheck *yourself* that all SMILES '
                'tokens are known to your SMILESLanguage object.'
            )

        if len(self.language_transforms.transforms) > 0:

            invalid_molecules = []
            for index in range(len(self._dataset)):
                self.smiles_language.add_smiles(
                    self.language_transforms(self._dataset[index])
                )

                if Chem.MolFromSmiles(self._dataset[index]) is None:
                    invalid_molecules.append(index)
            # Raise warning about invalid molecules
            if len(invalid_molecules) > 0:
                print(
                    f'NOTE: We found {len(invalid_molecules)} invalid smiles. '
                    'Check the warning trace. We recommend using '
                    'pytoda.smiles.smi_cleaner to remove the invalid SMILES in'
                    ' your .smi file.'
                )

            # Raise warning if new tokens were added.
            if len(self.smiles_language.token_to_index) > num_tokens:
                print(
                    f'{len(self.smiles_language.token_to_index) - num_tokens}'
                    ' new token(s) were added to SMILES language.'
                )

        transforms = language_transforms.copy()
        transforms += [
            SMILESToTokenIndexes(smiles_language=self.smiles_language)
        ]
        if self.randomize:
            transforms += [Randomize()]
        if self.padding:
            if padding_length is None:
                self.padding_length = (
                    self.smiles_language.max_token_sequence_length
                )
            transforms += [
                LeftPadding(
                    padding_length=self.padding_length,
                    padding_index=self.smiles_language.padding_index
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
