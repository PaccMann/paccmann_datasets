"""SMILES language handling."""
import dill
from collections import Counter
from ..files import read_smi
from ..types import FileList, Indexes, SMILESTokenizer, Tokens
from .processing import tokenize_smiles, SMILES_TOKENIZER


class SMILESLanguage(object):
    """
    SMILESLanguage class.

    SMILESLanguage handle SMILES data defining the vocabulary and
    utilities to manipulate it.
    """

    def __init__(
        self,
        name: str = 'smiles-language',
        smiles_tokenizer: SMILESTokenizer = (
            lambda smiles: tokenize_smiles(smiles, regexp=SMILES_TOKENIZER)
        ),
        add_start_and_stop: bool = False
    ) -> None:
        """
        Initialize SMILES language.

        Args:
            name (str): name of the SMILESLanguage.
            smiles_tokenizer (SMILESTokenizer): SMILES tokenization function.
                Defaults to tokenize_smiles.
            add_start_and_stop (bool): add <START> and <STOP> in the sequence,
                of tokens. Defaults to False.
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
        # NOTE: include augmentation characters, paranthesis and numbers for
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
            for index, token in additional_indexes_to_token.items()
        }
        self.add_start_and_stop = add_start_and_stop
        self._smiles_tokenizer_regexp = SMILES_TOKENIZER
        if self.add_start_and_stop:
            self.max_token_sequence_length = 2
            self._get_total_number_of_tokens_fn = (
                lambda tokens: len(tokens) + 2
            )
            self._finalize_token_indexes_fn = (
                lambda token_indexes:
                ([self.start_index] + token_indexes + [self.stop_index])
            )
        else:
            self.max_token_sequence_length = 0
            self._get_total_number_of_tokens_fn = len
            self._finalize_token_indexes_fn = (
                lambda token_indexes: token_indexes
            )

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

    def _update_max_token_sequence_length(self, tokens: Tokens) -> None:
        """
        Update the max token sequence length handling optional start and stop.

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
        for chunk in read_smi(smi_filepath, chunk_size=chunk_size):
            for smiles in chunk['SMILES']:
                self.add_smiles(smiles)

    def add_smiles(self, smiles: str) -> None:
        """
        Add a SMILES to the language.

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
        return self._finalize_token_indexes_fn(
            [
                self.token_to_index[token]
                for token in self.smiles_tokenizer(smiles)
                if token in self.token_to_index
            ]
        )

    def token_indexes_to_smiles(self, token_indexes: Indexes) -> str:
        """
        Transform a sequence of token indexes into SMILES.

        Args:
            token_indexes (Indexes): a sequence of token indexes.

        Returns:
            str: a SMILES representation.
        """
        return ''.join(
            [
                self.index_to_token.get(token_index, '')
                for token_index in token_indexes
                # consider only valid SMILES token indexes
                if token_index > 3
            ]
        )
