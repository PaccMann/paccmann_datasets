# Copyright 2020 Matteo Manica, Jannis Born, Ali Oskooei, Joris Cadow
# Most parts of this file are Licenced under the MIT Licence.
# Specifically the functions from_pretrained and save_pretrained are derivative
# works with sources under the following licence:
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use these functions except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SMILES language handling."""
import copy
import json
import logging
import os
import warnings
from collections import Counter

import rdkit  # Needs import before torch in some envs
import dill
import torch
from selfies import decoder as selfies_decoder
from selfies import encoder as selfies_encoder

from pytoda.warnings import device_warning

from ..files import read_smi
from ..transforms import Compose
from ..types import (
    Any,
    Files,
    Indexes,
    Iterable,
    Sequence,
    Tensor,
    Tokenizer,
    Tokens,
    Tuple,
    Union,
)
from .processing import TOKENIZER_FUNCTIONS, tokenize_smiles
from .transforms import compose_encoding_transforms, compose_smiles_transforms

logger = logging.getLogger(__name__)

# mimicry of huggingface tokenizers
# see PreTrainedTokenizer
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.json'}
# see PreTrainedTokenizerBase
TOKENIZER_CONFIG_FILE = 'tokenizer_config.json'
# our
TOKEN_COUNTS_FILE = 'token_count.json'


class UnknownMaxLengthError(RuntimeError):
    pass


class SMILESLanguage(object):
    """
    SMILESLanguage class.

    SMILESLanguage handle SMILES data defining the vocabulary and
    utilities to manipulate it, including encoding to token indexes.
    """

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        name: str = 'smiles-language',
        smiles_tokenizer: Tokenizer = tokenize_smiles,
        tokenizer_name: str = None,  # Literal only in python 3.8
        vocab_file: str = None,
        max_token_sequence_length: int = 0,
    ) -> None:
        """
        Initialize SMILES language.

        Args:
            name (str): name of the SMILESLanguage.
            smiles_tokenizer (Tokenizer): optional SMILES tokenization
                function. Defaults to tokenize_smiles, but tokenizer_name takes
                precedence when found in available TOKENIZER_FUNCTIONS.
            tokenizer_name (str): name, mapping to Tokenizer used to save and
                restore object from text files. Defaults to None, i.e.
                using default smiles_tokenizer. Examples for available names
                are 'smiles', 'selfies' or 'spe_smiles'.
            vocab_file (str): optional filepath to vocab json or directory
                containing it.
            max_token_sequence_length (int): initial value for keeping track
                of longest sequence. Defaults to 0.
        """
        self.name = name
        self.tokenizer_name = tokenizer_name
        if tokenizer_name is not None and tokenizer_name not in TOKENIZER_FUNCTIONS:
            logger.info(
                f'Given tokenizer_name {tokenizer_name} was not found, using'
                'default tokenizer function.'
            )
        self.smiles_tokenizer = TOKENIZER_FUNCTIONS.get(
            tokenizer_name, smiles_tokenizer
        )

        self.padding_token = '<PAD>'
        self.unknown_token = '<UNK>'
        self.start_token = '<START>'
        self.stop_token = '<STOP>'
        self.padding_index = 0
        self.unknown_index = 1
        self.start_index = 2
        self.stop_index = 3
        self.token_count = Counter()
        self.special_indexes = {
            self.padding_index: self.padding_token,
            self.unknown_index: self.unknown_token,
            self.start_index: self.start_token,
            self.stop_index: self.stop_token,
        }
        self.setup_vocab()

        if vocab_file:
            self.load_vocabulary(vocab_file)

        # updated when adding smiles
        self.max_token_sequence_length = max_token_sequence_length
        # updated by transformations, e.g. padding
        self._get_total_number_of_tokens_fn = len

        # inputs and kwargs for saving and re-loading (TOKENIZER_CONFIG_FILE)
        # (see ``from_pretrained`` and ``save_pretrained``)
        self.init_inputs = ()
        self.init_kwargs = {
            'name': self.name,
            'tokenizer_name': self.tokenizer_name,
            'max_token_sequence_length': self.max_token_sequence_length,
        }

        self.transform_smiles = Compose([])  # identity
        self.transform_encoding = Compose([])

    def setup_vocab(self) -> None:
        """
        Sets up the vocab by generating the special tokens.
        """
        # NOTE: include augmentation characters, parenthesis and numbers for
        #    rings
        additional_indexes_to_token = dict(
            enumerate(
                list('()')
                + list(map(str, range(1, 10)))
                + list('%{}'.format(index) for index in range(10, 30)),
                start=len(self.special_indexes),
            )
        )
        self.index_to_token = {**self.special_indexes, **additional_indexes_to_token}
        self.number_of_tokens = len(self.index_to_token)
        self.token_to_index = {
            token: index for index, token in self.index_to_token.items()
        }

    @staticmethod
    def load(filepath: str) -> 'SMILESLanguage':
        """
        Static method to load a SMILESLanguage object.

        Args:
            filepath (str): path to the file.

        Returns:
            SMILESLanguage: the loaded SMILES language object.
        """
        warnings.warn(
            "Loading languages will use a text files in the future", FutureWarning
        )
        try:
            with open(filepath, 'rb') as f:
                smiles_language = dill.load(f)
        except TypeError:
            # Necessary to load python3.7 pickled objects with >=3.8:
            # For details see: https://github.com/uqfoundation/dill/pull/406
            storage = dill._dill._reverse_typemap['CodeType']
            dill._dill._reverse_typemap['CodeType'] = dill._dill._create_code
            with open(filepath, 'rb') as f:
                smiles_language = dill.load(f)
            dill._dill._reverse_typemap['CodeType'] = storage
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
        warnings.warn(
            "Saving languages will only store a text files in the future", FutureWarning
        )
        SMILESLanguage.dump(self, filepath)

    def load_vocabulary(self, vocab_file: str):
        """Load a vocabulary mapping from token to token indexes.

        Args:
            vocab_file (str): a .json with tokens mapping to index. Can also
                be path to directory.
        """
        if os.path.isdir(vocab_file):
            vocab_file = os.path.join(vocab_file, self.vocab_files_names['vocab_file'])

        with open(vocab_file, encoding="utf-8") as fp:
            vocab = json.load(fp)
        # encoder
        self.token_to_index = self._check_specials(vocab)
        # decoder
        self.index_to_token = {v: k for k, v in self.token_to_index.items()}
        self.number_of_tokens = len(self.index_to_token)

    def _check_specials(self, vocab):
        """Check that defined special tokens match class definitions."""
        for index, token in self.special_indexes.items():
            try:
                if vocab[token] != index:
                    warnings.warn(
                        f'The vocab does not have matching special tokens: '
                        f'{token} is {vocab[token]}, but was defined as '
                        f'{index}.',
                    )
            except KeyError:
                warnings.warn(f'The vocab is missing a special token: {token}.')
        return vocab

    @classmethod
    def from_pretrained(cls, pretrained_path, *init_inputs, **kwargs):
        # directory with vocab files
        # not handling ADDED_TOKENS_FILE or SPECIAL_TOKENS_MAP_FILE
        # only handle case of files on disk here
        # but include handling optional counts
        resolved_vocab_files = {}

        additional_files_names = {
            'tokenizer_config_file': TOKENIZER_CONFIG_FILE,
            'token_count_file': TOKEN_COUNTS_FILE,
        }

        # Look for the tokenizer main vocabulary files
        # and the additional tokens files
        if os.path.isdir(pretrained_path):
            for file_id, file_name in {
                **cls.vocab_files_names,
                **additional_files_names,
            }.items():
                full_file_name = os.path.join(pretrained_path, file_name)
                if not os.path.exists(full_file_name):
                    logger.info(
                        "Didn't find file {}. We won't load it.".format(full_file_name)
                    )
                    full_file_name = None

                resolved_vocab_files[file_id] = full_file_name

        # Prepare tokenizer initialization kwargs
        tokenizer_config_file = resolved_vocab_files.pop('tokenizer_config_file', None)
        if tokenizer_config_file is not None:
            with open(tokenizer_config_file, encoding='utf-8') as config_file:
                init_kwargs = json.load(config_file)
            saved_init_inputs = init_kwargs.pop("init_inputs", ())
            if not init_inputs:
                init_inputs = saved_init_inputs
        else:
            init_kwargs = {}

        # Update with newly provided kwargs
        init_kwargs.update(kwargs)

        token_count_file = resolved_vocab_files.pop("token_count_file", None)

        # adds remaining (i.e. vocab_file) to kwargs
        for args_name, file_path in resolved_vocab_files.items():
            if args_name not in init_kwargs:
                init_kwargs[args_name] = file_path

        # Instantiate tokenizer.
        try:
            tokenizer = cls(*init_inputs, **init_kwargs)
        except OSError:
            raise OSError(
                'Unable to load vocabulary from file. '
                'Please check that the provided vocabulary is accessible '
                'and not corrupted.'
            )
        if token_count_file is not None:
            with open(token_count_file, encoding='utf-8') as counts_file:
                tokenizer.token_count = Counter(json.load(counts_file))

        # set args and kwargs explicitly here.
        tokenizer.init_inputs = init_inputs
        tokenizer.init_kwargs = init_kwargs

        return tokenizer

    def save_vocabulary(self, vocab_file: str) -> Tuple[str]:
        """Save the vocabulary mapping tokens to indexes to file.

        Args:
            vocab_file (str): a .json to save tokens mapping to index. Can also
                be path to directory.
        """
        if os.path.isdir(vocab_file):
            vocab_file = os.path.join(vocab_file, self.vocab_files_names['vocab_file'])

        with open(vocab_file, 'w', encoding="utf-8") as fp:
            json.dump(self.token_to_index, fp, indent=4)

        return (vocab_file,)

    def save_pretrained(self, save_directory):
        """Save the tokenizer vocabulary files together with
        tokenizer instantiation positional and keywords inputs.

        This method make sure the full tokenizer can then be re-loaded
        using the `from_pretrained` class method.
        """
        if not os.path.isdir(save_directory):
            # TODO raise?
            logger.error(
                'Saving directory ({}) should be a directory'.format(save_directory)
            )
            return

        tokenizer_config_file = os.path.join(save_directory, TOKENIZER_CONFIG_FILE)
        tokenizer_counts_file = os.path.join(save_directory, TOKEN_COUNTS_FILE)

        tokenizer_config = copy.deepcopy(self.init_kwargs)
        if len(self.init_inputs) > 0:
            tokenizer_config['init_inputs'] = copy.deepcopy(self.init_inputs)
        for file_id in self.vocab_files_names.keys():
            tokenizer_config.pop(file_id, None)

        with open(tokenizer_config_file, 'w', encoding='utf-8') as fp:
            json.dump(tokenizer_config, fp=fp, ensure_ascii=False, indent=4)

        with open(tokenizer_counts_file, 'w', encoding='utf-8') as fp:
            json.dump(self.token_count, fp=fp, ensure_ascii=False, indent=4)

        vocab_files = self.save_vocabulary(save_directory)

        return vocab_files + (tokenizer_counts_file,)

    def _load_vocabulary_from_pickled_language(
        self, filepath: str, include_metadata: bool = False
    ) -> None:
        """Save the vocabulary mapping tokens to indexes from file.

        Args:
            filepath (str): path to the dump of the SMILESLanguage.
        """
        a_language = self.load(filepath)
        # encoder
        self.token_to_index = self._check_specials(a_language.token_to_index)
        # decoder
        self.index_to_token = {v: k for k, v in self.token_to_index.items()}
        self.number_of_tokens = len(self.index_to_token)

    def _from_legacy_pickled_language(self, filepath: str) -> None:
        """Load a current language instance from pickled legacy language.

        Args:
            filepath (str): path to the dump of the SMILESLanguage.
        """
        warnings.warn(
            "Loading from legacy languages will be deprecated", DeprecationWarning
        )
        a_language = self.load(filepath)
        # encoder
        # missing special tokens
        self.token_to_index = a_language.token_to_index
        self.token_to_index.update({t: i for i, t in self.special_indexes.items()})
        # decoder
        self.index_to_token = {v: k for k, v in self.token_to_index.items()}
        self.number_of_tokens = len(self.index_to_token)

        self.max_token_sequence_length = a_language.max_token_sequence_length  # noqa
        self.init_kwargs['max_token_sequence_length'] = self.max_token_sequence_length
        self.token_count = a_language._token_count

    def _update_max_token_sequence_length(self, tokens: Tokens) -> None:
        """
        Update the max token sequence length.
        Uses method possibly overloaded by transformation setup to assess the
        length of tokens after transformations prior to their application.
        For example this allows handling start and stop tokens.

        Args:
            tokens (Tokens): tokens considered.
        """
        total_number_of_tokens = self._get_total_number_of_tokens_fn(tokens)
        if total_number_of_tokens > self.max_token_sequence_length:
            self.max_token_sequence_length = total_number_of_tokens
            self.init_kwargs['max_token_sequence_length'] = total_number_of_tokens

    def _update_language_dictionaries_with_tokens(self, tokens: Tokens) -> None:
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
                self.number_of_tokens,
            )
        )
        # update language
        self.token_count += tokens_counter
        self.index_to_token.update(index_to_token)
        self.token_to_index.update(
            {token: index for index, token in index_to_token.items()}
        )
        self.number_of_tokens += len(index_to_token)

    def add_smis(
        self,
        smi_filepaths: Files,
        index_col: int = 1,
        chunk_size: int = 10000,
        name: str = 'SMILES',
        names: Sequence[str] = None,
    ) -> None:
        """
        Add a set of SMILES from a list of .smi files, applying
        `transform_smiles`.

        Args:
            smi_filepaths (Files): a list of paths to .smi files.
            index_col (int): Data column used for indexing, defaults to 1.
            chunk_size (int): size of the chunks. Defaults to 10000.
            name (str): type of dataset, used to index columns in smi, and must
                be in names. Defaults to 'SMILES'.
            names (Sequence[str]): User-assigned names given to the columns.
                Defaults to `[name]`.
        """
        for smi_filepath in smi_filepaths:
            self.add_smi(
                smi_filepath,
                index_col=index_col,
                chunk_size=chunk_size,
                name=name,
                names=names,
            )

    def add_smi(
        self,
        smi_filepath: str,
        index_col: int = 1,
        chunk_size: int = 10000,
        name: str = 'SMILES',
        names: Sequence[str] = None,
    ) -> None:
        """
        Add a set of SMILES from a .smi file, applying `transform_smiles`.

        Args:
            smi_filepath (str): path to the .smi file.
            index_col (int): Data column used for indexing, defaults to 1.
            chunk_size (int): number of rows to read in a chunk.
                Defaults to 100000.
            name (str): type of dataset, used to index columns in smi, and must
                be in names. Defaults to 'SMILES'.
            names (Sequence[str]): User-assigned names given to the columns.
                Defaults to `[name]`.
        """
        names = names or [name]
        try:
            for chunk in read_smi(
                smi_filepath, index_col=index_col, chunk_size=chunk_size, names=names
            ):
                for smiles in chunk[name]:
                    try:
                        transformed_smiles = self.transform_smiles(smiles)
                        self.add_smiles(transformed_smiles)
                    except Exception:
                        logger.warning(
                            'transformation of smiles or adding result to '
                            f'the language failed for: {smiles}'
                        )
        except IndexError:
            raise IndexError('There must be one name per column in names.')
        except KeyError as error:
            raise KeyError(
                f'{str(error)}. Check index_col and that name {name} is in '
                f' names {names}'
            )

    def add_dataset(self, dataset: Iterable):
        """
        Add a set of SMILES from an iterable, applying `transform_smiles`.

        Collects and warns about invalid SMILES, and warns on finding new
        tokens.

        Args:
            dataset (Iterable): returning SMILES strings.
        """
        initial_vocab_length = len(self.token_to_index)
        self.invalid_molecules = []
        self.failed_transform_smiles = []

        for index, smiles in enumerate(dataset):
            if rdkit.Chem.MolFromSmiles(smiles, sanitize=False) is None:
                self.invalid_molecules.append((index, smiles))
            else:
                try:
                    transformed_smiles = self.transform_smiles(smiles)
                except Exception:
                    self.failed_transform_smiles.append((index, smiles))
                else:
                    self.add_smiles(transformed_smiles)

        # Raise warning about invalid molecules
        if len(self.invalid_molecules) > 0:
            logger.warning(
                f'NOTE: We found {len(self.invalid_molecules)} invalid  '
                'smiles. Check the warning trace and inspect the  attribute '
                '`invalid_molecules`. To remove invalid  SMILES in your .smi '
                'file, we recommend using '
                '`pytoda.preprocessing.smi.smi_cleaner`.'
            )
        # Raise warning about failed transformations
        if len(self.failed_transform_smiles) > 0:
            logger.warning(
                f'NOTE: We found {len(self.failed_transform_smiles)} smiles '
                'that failed to be transformed (excluding invalid smiles). '
                'Inspect the attribute `failed_transform_smiles`.'
            )

        # Raise warning if new tokens were added.
        if len(self.token_to_index) > initial_vocab_length:
            logger.warning(
                f'{len(self.token_to_index) - initial_vocab_length}'
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
            self.token_count[token] += 1
        else:
            self.token_to_index[token] = self.number_of_tokens
            self.token_count[token] = 1
            self.index_to_token[self.number_of_tokens] = token
            self.number_of_tokens += 1

    def smiles_to_token_indexes(self, smiles: str) -> Union[Indexes, Tensor]:
        """
        Transform character-level SMILES into a sequence of token indexes.

        Args:
            smiles (str): a SMILES (or SELFIES) representation.

        Returns:
            Union[Indexes, Tensor]: indexes representation for the
                SMILES/SELFIES provided.
        """
        return self.transform_encoding(
            [
                self.token_to_index.get(token, self.unknown_index)
                for token in self.smiles_tokenizer(self.transform_smiles(smiles))
            ]
        )

    def token_indexes_to_smiles(self, token_indexes: Union[Indexes, Tensor]) -> str:
        """
        Transform a sequence of token indexes into SMILES, ignoring special
        tokens.

        Args:
            token_indexes (Union[Indexes, Tensor]): Sequence of integers
                representing tokens in vocabulary.

        Returns:
            str: a SMILES (or SELFIES) representation.
        """
        token_indexes = self.tensor_to_indexes(token_indexes)

        return ''.join(
            [
                self.index_to_token.get(token_index, '')
                for token_index in token_indexes
                # consider only valid SMILES token indexes
                if token_index not in self.special_indexes
            ]
        )

    @staticmethod
    def tensor_to_indexes(token_indexes: Union[Indexes, Tensor]) -> Indexes:
        """Utility to get Indexes from Tensors.

        Args:
            token_indexes (Union[Indexes, Tensor]): from single SMILES.

        Raises:
            ValueError: in case the Tensor is not shaped correctly

        Returns:
            Indexes: list from Tensor or else the initial token_indexes.
        """
        if isinstance(token_indexes, torch.Tensor):
            if token_indexes.ndim != 1:
                raise ValueError('Only token indexes for a single SMILES are supported')
            return token_indexes.numpy().flatten().tolist()

        return token_indexes

    def selfies_to_smiles(self, selfies: str) -> str:
        """
        SELFIES to SMILES converter method.
        Based on: https://arxiv.org/abs/1905.13741

        Arguments:
            selfies (str): SELFIES representation

        Returns:
            str: A SMILES string
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
            smiles (str): SMILES representation

        Returns:
            str: A SELFIES string
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


class SELFIESLanguage(SMILESLanguage):
    """
    SELFIESLanguage is a SMILESLanguage with a different default tokenizer,
    transforming SMILES to SELFIES.
    """

    def __init__(
        self,
        name: str = 'selfies-language',
        vocab_file: str = None,
        max_token_sequence_length: int = 0,
    ) -> None:
        """
        Initialize SMILES language.

        Args:
            name (str): name of the SMILESLanguage.
            vocab_file (str): optional filepath to vocab json or directory
                containing it.
            max_token_sequence_length (int): initial value for keeping track
                of longest sequence. Defaults to 0.
        """
        super().__init__(
            name=name,
            tokenizer_name='selfies',
            vocab_file=vocab_file,
            max_token_sequence_length=max_token_sequence_length,
        )
        self.transform_smiles = selfies_encoder


class SMILESTokenizer(SMILESLanguage):
    """
    SMILESTokenizer class, based on SMILESLanguage applying transforms and
    and encoding of SMILES string to sequence of token indexes.
    """

    def __init__(
        self,
        name: str = 'smiles-language',
        smiles_tokenizer: Tokenizer = tokenize_smiles,
        tokenizer_name: str = None,
        vocab_file: str = None,
        max_token_sequence_length: int = 0,
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
        padding: bool = False,
        padding_length: int = None,
        device: Any = None,
    ) -> None:
        """
        Initialize SMILES language.

        Args:
            name (str): name of the SMILESLanguage.
            smiles_tokenizer (Tokenizer): optional SMILES tokenization
                function. Defaults to tokenize_smiles, but tokenizer_name takes
                precedence when found in available TOKENIZER_FUNCTIONS.
            tokenizer_name (str): optional name mapping to Tokenizer. Defaults
                to None, i.e. using default smiles_tokenizer.
            vocab_file (str): optional filepath to vocab json or directory
                containing it.
            max_token_sequence_length (int): initial value for keeping track
                of longest sequence. Defaults to 0.
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
            padding (bool): pad sequences from the left to matching length.
                Defaults to False.
            padding_length (int): common length of all token sequences,
                applies only if padding is True. See `set_max_padding` to set
                it to longest token sequence the smiles language encountered.
                Defaults to None.
            device (Any): Deprecated argument that will be removed in the future.

        NOTE:
            See `set_smiles_transforms` and `set_encoding_transforms` to change
            the transforms temporarily and reset with
            `reset_initial_transforms`. Assignment of class attributes
            in the parameter list will trigger such a reset.
        """
        device_warning(device)
        super().__init__(
            name=name,
            smiles_tokenizer=smiles_tokenizer,
            tokenizer_name=tokenizer_name,
            vocab_file=vocab_file,
            max_token_sequence_length=max_token_sequence_length,
        )
        # smiles transforms
        self.canonical = canonical
        self.augment = augment
        self.kekulize = kekulize
        self.all_bonds_explicit = all_bonds_explicit
        self.all_hs_explicit = all_hs_explicit
        self.remove_bonddir = remove_bonddir
        self.remove_chirality = remove_chirality
        self.selfies = selfies
        self.sanitize = sanitize
        # encoding transforms
        self.randomize = randomize
        self.add_start_and_stop = add_start_and_stop
        self.padding = padding
        self.padding_length = padding_length

        if device:
            logger.warning(device_warning(device))

        self._init_attributes = [  # additions to init_kwargs for pretrained
            'canonical',
            'augment',
            'kekulize',
            'all_bonds_explicit',
            'all_hs_explicit',
            'remove_bonddir',
            'remove_chirality',
            'selfies',
            'sanitize',
            'randomize',
            'add_start_and_stop',
            'padding',
            'padding_length',
        ]
        # update save/load pretrained kwargs
        for keyword in self._init_attributes:
            self.init_kwargs[keyword] = getattr(self, keyword)

        self.reset_initial_transforms()

        self._attributes_to_trigger_reset = [
            *self._init_attributes,
            'start_index',
            'stop_index',
        ]  # could be updated in inheritance

        # only now 'activate' setter that resets the transforms and warns on
        # truncating padding_length
        self._initialized = True

    def __setattr__(self, name, value):
        """Also updates the transforms if the set attribute affects them."""
        super().__setattr__(name, value)
        if self.__dict__.get('_initialized'):
            if name in self._attributes_to_trigger_reset:
                self.reset_initial_transforms()
                if name in self._init_attributes:
                    self.init_kwargs[name] = value
            if name == 'padding_length' and self.padding:
                if self.max_token_sequence_length > value:
                    logger.warning(
                        'The language has seen sequences of length '
                        f'{self.max_token_sequence_length} that will be '
                        'truncated by given padding length of '
                        f'{value}. Consider `set_max_padding`.'
                    )

    @staticmethod
    def __get_total_number_of_tokens_with_start_stop_fn(x):
        return len(x) + 2

    def _set_token_len_fn(self, add_start_and_stop):
        """
        Defines a Callable that given a sequence of naive tokens, i.e. before
        applying the encoding transforms, computes the number of
        implicit tokens after transforms (implicit because it's the
        number of token indexes, not actual tokens).
        """
        if add_start_and_stop:
            self._get_total_number_of_tokens_fn = (
                self.__get_total_number_of_tokens_with_start_stop_fn
            )
        else:
            self._get_total_number_of_tokens_fn = len

    def set_max_padding(self):
        """
        Set padding_length that does not truncate any sequence. Requires
        updated max_token_sequence_length.

        Raises:
            UnknownMaxLengthError: When max_token_sequence_length is 0 because
                no SMILES were added to the language.
        """
        if self.max_token_sequence_length == 0:
            raise UnknownMaxLengthError(
                'No check possible for naive SMILESTokenizer. Instance needs '
                'a pass over the data, setting max_token_sequence_length. '
                'See for example `add_smis`, `add_dataset` or `add_smiles` '
                'methods.'
            )

        # also triggers reset of transforms
        self.padding_length = self.max_token_sequence_length

    def reset_initial_transforms(self):
        """Reset smiles and token indexes transforms as on initialization."""
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
        self.transform_encoding = compose_encoding_transforms(
            self.randomize,
            self.add_start_and_stop,
            self.start_index,
            self.stop_index,
            self.padding,
            self.padding_length,
            self.padding_index,
        )
        self._set_token_len_fn(self.add_start_and_stop)

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
        """Helper function to reversibly change steps of the transforms."""
        self.transform_smiles = compose_smiles_transforms(
            canonical=canonical if canonical is not None else self.canonical,
            augment=augment if augment is not None else self.augment,
            kekulize=kekulize if kekulize is not None else self.kekulize,
            all_bonds_explicit=all_bonds_explicit
            if all_bonds_explicit is not None
            else self.all_bonds_explicit,
            all_hs_explicit=all_hs_explicit
            if all_hs_explicit is not None
            else self.all_hs_explicit,
            remove_bonddir=remove_bonddir
            if remove_bonddir is not None
            else self.remove_bonddir,
            remove_chirality=remove_chirality
            if remove_chirality is not None
            else self.remove_chirality,
            selfies=selfies if selfies is not None else self.selfies,
            sanitize=sanitize if sanitize is not None else self.sanitize,
        )

    def set_encoding_transforms(
        self,
        randomize=None,
        add_start_and_stop=None,
        padding=None,
        padding_length=None,
    ):
        """Helper function to reversibly change steps of the transforms."""
        self.transform_encoding = compose_encoding_transforms(
            randomize=randomize if randomize is not None else self.randomize,
            add_start_and_stop=add_start_and_stop
            if add_start_and_stop is not None
            else self.add_start_and_stop,
            start_index=self.start_index,
            stop_index=self.stop_index,
            padding=padding if padding is not None else self.padding,
            padding_length=padding_length
            if padding_length is not None
            else self.padding_length,
            padding_index=self.padding_index,
        )
        if add_start_and_stop is not None:
            self._set_token_len_fn(add_start_and_stop)
