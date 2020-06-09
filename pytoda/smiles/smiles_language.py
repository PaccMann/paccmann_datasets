"""SMILES language handling."""
import json
import logging
from collections import Counter

import dill
import torch
from rdkit import Chem
from selfies import decoder as selfies_decoder
from selfies import encoder as selfies_encoder

from ..files import read_smi
from ..types import FileList, Indexes, Iterable, SMILESTokenizer, Tokens
from .processing import SMILES_TOKENIZER, tokenize_smiles
from .transforms import compose_encoding_transforms, compose_smiles_transforms
from ..transforms import Compose

logger = logging.getLogger(__name__)


class SMILESLanguage(object):
    """
    SMILESLanguage class.

    SMILESLanguage handle SMILES data defining the vocabulary and
    utilities to manipulate it, including encoding to token indexes.
    """

    def __init__(
        self,
        name: str = 'smiles-language',
        smiles_tokenizer: SMILESTokenizer = (
            lambda smiles: tokenize_smiles(smiles, regexp=SMILES_TOKENIZER)
        ),
        **kwargs
    ) -> None:
        """
        Initialize SMILES language.

        Args:
            name (str): name of the SMILESLanguage.
            smiles_tokenizer (SMILESTokenizer): SMILES tokenization function.
                Defaults to tokenize_smiles.
            kwargs (dict): ignoring additional arguments.
        """
        self.name = name
        self.smiles_tokenizer = smiles_tokenizer
        self.padding_token = '<PAD>'
        self.unknown_token = '<UNK>'
        self.start_token = '<START>'
        self.stop_token = '<STOP>'
        self.padding_index = 0
        self.unknown_index = 1
        self.start_index = 2
        self.stop_index = 3
        self._token_count = Counter()
        self.index_to_token = {
            self.padding_index: self.padding_token,
            self.unknown_index: self.unknown_token,
            self.start_index: self.start_token,
            self.stop_index: self.stop_token,
        }
        # NOTE: include augmentation characters, parenthesis and numbers for
        #    rings
        additional_indexes_to_token = dict(
            enumerate(
                list('()') + list(map(str, range(1, 10))) +
                list('%{}'.format(index) for index in range(10, 30)),
                start=len(self.index_to_token)
            )
        )
        self.index_to_token.update(additional_indexes_to_token)
        self.number_of_tokens = len(self.index_to_token)
        self.token_to_index = {
            token: index
            for index, token in self.index_to_token.items()
        }

        # updated when adding smiles
        self.max_token_sequence_length = 0
        # updated by transformations, e.g. padding
        self._get_total_number_of_tokens_fn = len
        self.transform_smiles = Compose([])  # identity
        self.transform_encoding = Compose([])

    @staticmethod
    def load(filepath: str) -> 'SMILESLanguage':
        """
        Static method to load a SMILESLanguage object.

        Args:
            filepath (str): path to the file.

        Returns:
            SMILESLanguage: the loaded SMILES language object.
        """
        with open(filepath, 'rb') as f:
            smiles_language = dill.load(f)
        return smiles_language

    @staticmethod
    def dump(smiles_language: 'SMILESLanguage', filepath: str):
        """
        Static method to save a smiles_language object to disk.

        Args:
            smiles_language (SMILESLanguage): a SMILESLanguage object.
            filepath (str): path where to dump the SMILESLanguage.
        """
        with open(filepath, 'wb') as f:
            dill.dump(smiles_language, f)

    def save(self, filepath: str):
        """
        Instance method to save/dump smiles language object.

        Args:
            filepath (str): path where to save the SMILESLanguage.
        """
        SMILESLanguage.dump(self, filepath)

    def load_vocab(self, vocab_file: str):
        """Load a vocabulary mapping from token to token indices."""
        # encoder
        with open(vocab_file, encoding="utf-8") as fp:
            self.token_to_index = json.load(fp)
        # decoder
        self.index_to_token = {v: k for k, v in self.token_to_index.items()}

    def save_vocab(self, vocab_file: str):
        with open(vocab_file, 'w', encoding="utf-8") as fp:
            json.dump(self.token_to_index, fp)

    def _load_counts(self, counts_file: str):
        """Restore token counts stored from a prior smiles_language."""
        with open(counts_file, encoding="utf-8") as fp:
            self._token_count = Counter(
                json.load(fp)
            )

    def _save_counts(self, counts_file: str):
        with open(counts_file, 'w', encoding="utf-8") as fp:
            json.dump(self._token_count, fp)

    def _load_max_token_sequence_length(self, max_len_file: str):
        """Restore max length stored from a prior smiles_language."""
        with open(max_len_file, encoding="utf-8") as fp:
            self._max_token_sequence_length = int(fp.readline())

    def _save_max_token_sequence_length(self, max_len_file: str):
        with open(max_len_file, 'w', encoding="utf-8") as fp:
            fp.write(str(self._max_token_sequence_length))

    def _update_max_token_sequence_length(self, tokens: Tokens) -> None:
        """
        Update the max token sequence length.
        Uses method possibly overloaded by transformation to a

        Args:
            tokens (Tokens): tokens considered.
        """
        total_number_of_tokens = self._get_total_number_of_tokens_fn(tokens)
        if total_number_of_tokens > self.max_token_sequence_length:
            self.max_token_sequence_length = total_number_of_tokens

    def _update_language_dictionaries_with_tokens(
        self, tokens: Tokens
    ) -> None:
        """
        Update the language dictionaries with provided tokens.

        Args:
            tokens (Tokens): tokens considered.
        """
        # count tokens
        tokens_counter = Counter(tokens)
        # index to token
        index_to_token = dict(
            enumerate(
                tokens_counter.keys() - self.token_to_index.keys(),
                self.number_of_tokens
            )
        )
        # update language
        self._token_count += tokens_counter
        self.index_to_token.update(index_to_token)
        self.token_to_index.update(
            {token: index
             for index, token in index_to_token.items()}
        )
        self.number_of_tokens += len(index_to_token)

    def add_smis(
        self, smi_filepaths: FileList, chunk_size: int = 100000
    ) -> None:
        """
        Add a set of SMILES from a list of .smi files.

        Args:
            smi_filepaths (FileList): a list of paths to .smi files.
            chunk_size (int): number of rows to read in a chunk.
                Defaults to 100000.
        """
        for smi_filepath in smi_filepaths:
            self.add_smi(smi_filepath, chunk_size=chunk_size)

    def add_smi(self, smi_filepath: str, chunk_size: int = 100000) -> None:
        """
        Add a set of SMILES from a .smi file.

        Args:
            smi_filepath (str): path to the .smi file.
            chunk_size (int): number of rows to read in a chunk.
                Defaults to 100000.
        """
        try:
            for chunk in read_smi(smi_filepath, chunk_size=chunk_size):
                for smiles in chunk['SMILES']:
                    self.add_smiles(self.transform_smiles(smiles))
        except Exception:
            raise KeyError(
                ".smi file needs to have 2 columns, first with IDs, second "
                "with SMILES."
            )

    def add_dataset(self, dataset: Iterable):
        """
        Add a set of SMILES from an iterable.

        Collects and warns about invalid SMILES, and warns on finding new
        tokens.

        Args:
            dataset (Iterable): returning SMILES strings.
        """
        num_tokens = len(self.token_to_index)

        self.invalid_molecules = []
        for index, smiles in enumerate(dataset):
            smiles = self.transform_smiles(smiles)
            self.add_smiles(smiles)
            if Chem.MolFromSmiles(smiles) is None:  # fails e.g. for selfies
                self.invalid_molecules.append(tuple(index, smiles))
        # Raise warning about invalid molecules
        if len(self.invalid_molecules) > 0:
            logger.warning(
                f'NOTE: We found {len(self.invalid_molecules)} invalid  '
                'smiles. Check the warning trace and inspect the  attribute '
                '`invalid_molecules`. To remove invalid  SMILES in your .smi '
                'file, we recommend using '
                '`pytoda.preprocessing.smi.smi_cleaner`.'
            )

        # Raise warning if new tokens were added.
        if len(self.token_to_index) > num_tokens:
            logger.warning(
                f'{len(self.token_to_index) - num_tokens}'
                ' new token(s) were added to SMILES language.'
            )

    def add_smiles(self, smiles: str) -> None:
        """
        Add a SMILES to the language.

        Updates `max_token_sequence_length`.
        Adds missing tokens to the language.

        Args:
            smiles (str): a SMILES representation.
        """
        tokens = self.smiles_tokenizer(smiles)
        self._update_max_token_sequence_length(tokens)
        self._update_language_dictionaries_with_tokens(tokens)

    def add_token(self, token: str) -> None:
        """
        Add a token to the language.

        Args:
            token (str): a token.
        """
        if token in self.token_to_index:
            self._token_count[token] += 1
        else:
            self.token_to_index[token] = self.number_of_tokens
            self._token_count[token] = 1
            self.index_to_token[self.number_of_tokens] = token
            self.number_of_tokens += 1

    def smiles_to_token_indexes(self, smiles: str) -> Indexes:
        """
        Transform character-level SMILES into a sequence of token indexes.

        Args:
            smiles (str): a SMILES (or SELFIES) representation.

        Returns:
            Indexes: indexes representation for the SMILES/SELFIES provided.
        """
        return [
            self.transform_encoding(self.token_to_index[token])
            for token in self.smiles_tokenizer(smiles)
            if token in self.token_to_index  # TODO Warning or Fail?
        ]

    def token_indexes_to_smiles(self, token_indexes: Indexes) -> str:
        """
        Transform a sequence of token indexes into SMILES, ignoring special
        tokens.

        Args:
            token_indexes (Indexes): a sequence of token indexes.

        Returns:
            str: a SMILES (or SELFIES) representation.
        """
        return ''.join(
            [
                self.index_to_token.get(token_index, '')
                for token_index in token_indexes
                # consider only valid SMILES token indexes
                if token_index > 3
            ]
        )

    def selfies_to_smiles(self, selfies: str) -> str:
        """
        SELFIES to SMILES converter method.
        Based on: https://arxiv.org/abs/1905.13741

        Arguments:
            selfies {str} -- SELFIES representation

        Returns:
            str -- A SMILES string
        """
        if not isinstance(selfies, str):
            raise TypeError(f'Wrong data type: {type(selfies)}. Use strings.')
        try:
            return selfies_decoder(selfies)
        except Exception:
            logger.warning(
                f'Could not convert SELFIES {selfies} to SMILES, returning '
                'the SELFIES instead'
            )
            return selfies

    def smiles_to_selfies(self, smiles: str) -> str:
        """
        SMILES to SELFIES converter method.
        Based on: https://arxiv.org/abs/1905.13741

        Arguments:
            smiles {str} -- smiles representation

        Returns:
            str -- A SELFIES string
        """
        if not isinstance(smiles, str):
            raise TypeError(f'Wrong data type: {type(smiles)}. Use strings.')
        try:
            return selfies_encoder(smiles)
        except Exception:
            logger.warning(
                f'Could not convert SMILES {smiles} to SELFIES, returning '
                'the SMILES instead'
            )
            return smiles


class SMILESEncoder(SMILESLanguage):
    """
    SMILESEncoder class, based on SMILESLanguage applying transforms and
    and encoding of SMILES string to sequence of token indices.
    """

    def __init__(
        self,
        name: str = 'smiles-language',
        smiles_tokenizer: SMILESTokenizer = (
            lambda smiles: tokenize_smiles(smiles, regexp=SMILES_TOKENIZER)
        ),
        canonical: bool = False,  #
        augment: bool = False,
        kekulize: bool = False,
        all_bonds_explicit: bool = False,
        all_hs_explicit: bool = False,
        remove_bonddir: bool = False,
        remove_chirality: bool = False,
        selfies: bool = False,
        sanitize: bool = True,
        randomize: bool = False,  #
        add_start_and_stop: bool = False,
        padding: bool = True,
        padding_length: int = None,
        device: torch.device = torch.
            device('cuda' if torch.cuda.is_available() else 'cpu'),
    ) -> None:
        """
        Initialize SMILES language.

        Args:
            name (str): name of the SMILESLanguage.
            smiles_tokenizer (SMILESTokenizer): SMILES tokenization function.
                Defaults to tokenize_smiles.
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
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
        """
        super().__init__(self, name, smiles_tokenizer)

        self.canonical = canonical
        self.augment = augment
        self.kekulize = kekulize
        self.all_bonds_explicit = all_bonds_explicit
        self.all_hs_explicit = all_hs_explicit
        self.remove_bonddir = remove_bonddir
        self.remove_chirality = remove_chirality
        self.selfies = selfies
        self.sanitize = sanitize

        self.randomize = randomize
        self.add_start_and_stop = add_start_and_stop
        self.padding = padding
        self.padding_length = padding_length
        self.device = device

        if padding:
            self.padding_length = self._check_padding_length(padding_length)

        self.set_initial_transforms()

    def _check_padding_length(self, padding_length):
        if padding_length is None:
            if self.max_token_sequence_length == 0:
                raise ValueError(
                    'Padding with naive SMILESEncoder (no pass over data, '
                    'finding longest SMILES) requires passing explicit '
                    'padding_length argument.'
                )
            else:  # pad to known longest sequence
                padding_length = self.max_token_sequence_length

        if self.max_token_sequence_length > padding_length:
            logger.warning(
                f'From passing over the dataset the given padding length '
                f'was found to trunkate some sequence of max length '
                f'{self.max_token_sequence_length}.'
            )
        return padding_length

    def set_initial_transforms(self):
        """reset smiles and token indices transforms as on initialization."""
        self.transform_smiles = compose_smiles_transforms(
            self.canonical,
            self.augment,
            self.kekulize,
            self.all_bonds_explicit,
            self.all_hs_explicit,
            self.remove_bonddir,
            self.remove_chirality,
            self.selfies,
            self.sanitize,
        )
        (
            self.transform_encoding,
            self._get_total_number_of_tokens_fn
        ) = compose_encoding_transforms(
            self.randomize,
            self.add_start_and_stop,
            self.start_index,
            self.stop_index,
            self.padding,
            self.padding_length,
            self.padding_index,
            self.device,
        )

    def set_smiles_transforms(
        self,
        canonical=None,
        augment=None,
        kekulize=None,
        all_bonds_explicit=None,
        all_hs_explicit=None,
        remove_bonddir=None,
        remove_chirality=None,
        selfies=None,
        sanitize=None,
    ):
        """Helper function to change specific steps of the transforms."""
        self.transform_smiles = compose_smiles_transforms(
            canonical=canonical if canonical else self.canonical,
            augment=augment if augment else self.augment,
            kekulize=kekulize if kekulize else self.kekulize,
            all_bonds_explicit=all_bonds_explicit if all_bonds_explicit else self.all_bonds_explicit,  # noqa
            all_hs_explicit=all_hs_explicit if all_hs_explicit else self.all_hs_explicit,  # noqa
            remove_bonddir=remove_bonddir if remove_bonddir else self.remove_bonddir,  # noqa
            remove_chirality=remove_chirality if remove_chirality else self.remove_chirality,  # noqa
            selfies=selfies if selfies else self.selfies,
            sanitize=sanitize if sanitize else self.sanitize,
        )

    def set_encoding_transforms(
        self,
        randomize=None,
        add_start_and_stop=None,
        padding=None,
        padding_length=None,
        device=None,
    ):
        """Helper function to change specific steps of the transforms."""
        if padding:
            padding_length = self._check_padding_length(
                padding_length if padding_length else self.padding_length
            )
        self.transform_encodings = compose_encoding_transforms(
            randomize=randomize if randomize else self.randomize,
            add_start_and_stop=add_start_and_stop if add_start_and_stop else self.add_start_and_stop,  # noqa
            start_index=self.start_index,
            stop_index=self.stop_index,
            padding=padding if padding else self.padding,
            padding_length=padding_length,
            padding_index=self.padding_index,
            device=device if device else self.device,
        )
