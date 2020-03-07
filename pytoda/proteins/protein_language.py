"""Protein language handling."""

import dill
from ..files import read_smi
from ..types import Indexes, Tokens
from .processing import IUPAC_VOCAB, UNIREP_VOCAB


class ProteinLanguage(object):
    """
    ProteinLanguage class.

    ProteinLanguage handle Protein data defining the vocabulary and
    utilities to manipulate it.
    """

    def __init__(
        self,
        name: str = 'protein-language',
        amino_acid_dict: str = 'iupac',
        add_start_and_stop: bool = True
    ) -> None:
        """
        Initialize Protein language.

        Args:
            name (str): name of the ProteinLanguage.
            amino_acid_dict (str): Tokenization regime for amino acid
                sequence. Defaults to 'iupac', alternative is 'unirep'.
            add_start_and_stop (bool): add <START> and <STOP> in the sequence,
                of tokens. Defaults to True.
        """
        self.name = name
        self.dict = amino_acid_dict
        self.add_start_and_stop = add_start_and_stop

        if self.dict == 'iupac':
            self.token_to_index = IUPAC_VOCAB
        elif self.dict == 'unirep':
            self.token_to_index = UNIREP_VOCAB
        else:
            raise ValueError(
                "Choose dict as 'iupac' or 'unirep' (given was"
                f"{amino_acid_dict})."
            )
        # Setup dictionary
        self.sequence_tokens = [
            index for token, index in self.token_to_index.items()
            if '<' not in token
        ]

        self.tokenizer = list
        self.number_of_tokens = len(self.token_to_index)
        self.index_to_token = {
            index: token
            for token, index in self.token_to_index.items()
        }

        if self.add_start_and_stop:
            self.max_token_sequence_length = 2
            self._get_total_number_of_tokens_fn = (
                lambda tokens: len(tokens) + 2
            )
            self._finalize_token_indexes_fn = (
                lambda token_indexes: (
                    [self.token_to_index['<START>']] + token_indexes +
                    [self.token_to_index['<STOP>']]
                )
            )
        else:
            self.max_token_sequence_length = 0
            self._get_total_number_of_tokens_fn = len
            self._finalize_token_indexes_fn = (
                lambda token_indexes: token_indexes
            )

    def __len__(self) -> int:
        """Number of characters the language knows."""
        return self.number_of_tokens

    @staticmethod
    def load(filepath: str) -> 'ProteinLanguage':
        """
        Static method to load a ProteinLanguage object.

        Args:
            filepath (str): path to the file.

        Returns:
            ProteinLanguage: the loaded Protein language object.
        """
        with open(filepath, 'rb') as f:
            protein_language = dill.load(f)
        return protein_language

    @staticmethod
    def dump(protein_language: 'ProteinLanguage', filepath: str):
        """
        Static method to save a Protein_language object to disk.

        Args:
            protein_language (ProteinLanguage): a ProteinLanguage object.
            filepath (str): path where to dump the ProteinLanguage.
        """
        with open(filepath, 'wb') as f:
            dill.dump(protein_language, f)

    def save(self, filepath: str):
        """
        Instance method to save/dump Protein language object.

        Args:
            filepath (str): path where to save the ProteinLanguage.
        """
        ProteinLanguage.dump(self, filepath)

    def _update_max_token_sequence_length(self, tokens: Tokens) -> None:
        """
        Update the max token sequence length handling optional start and stop.

        Args:
            tokens (Tokens): tokens considered.
        """
        total_number_of_tokens = self._get_total_number_of_tokens_fn(tokens)
        if total_number_of_tokens > self.max_token_sequence_length:
            self.max_token_sequence_length = total_number_of_tokens

    def add_file(
        self, filepath: str, index_col: int = 1, chunk_size: int = 100000
    ) -> None:
        """
        Add a set of SMILES from a .smi file.

        Args:
            filepath (str): path to the .smi file.
            chunk_size (int): number of rows to read in a chunk.
                Defaults to 100000.
            index_col (int): Data column used for indexing, defaults to 1.
        """
        for chunk in read_smi(
            filepath,
            chunk_size=chunk_size,
            index_col=index_col,
            names=['Sequence']
        ):
            for sequence in chunk['Sequence']:
                self.add_sequence(sequence)

        try:
            for chunk in read_smi(
                filepath,
                chunk_size=chunk_size,
                index_col=index_col,
                names=['Sequence']
            ):
                for sequence in chunk['Sequence']:
                    self.add_sequence(sequence)
        except Exception:
            raise KeyError(
                ".smi file needs to have 2 columns, index needs to be in "
                f"column ({index_col}), sequences in the other."
            )

    def add_sequence(self, sequence: str) -> None:
        """
        Add a amino acid sequence to the language.

        Args:
            sequence (str): a sequence of amino acids.
        """
        tokens = self.tokenizer(sequence)
        self._update_max_token_sequence_length(tokens)

    def sequence_to_token_indexes(self, sequence: str) -> Indexes:
        """
        Transform character-level amino acid sequence (AAS) into a sequence of
        token indexes.

        Args:
            sequence (str): an AAS representation.

        Returns:
            Indexes: indexes representation for the AAS provided.
        """
        return self._finalize_token_indexes_fn(
            [
                self.token_to_index[token]
                for token in self.tokenizer(sequence)
                if token in self.token_to_index
            ]
        )

    def token_indexes_to_sequence(self, token_indexes: Indexes) -> str:
        """
        Transform a sequence of token indexes into amino acid sequence.

        Args:
            token_indexes (Indexes): a sequence of token indexes.

        Returns:
            str: an amino acid sequence representation.
        """
        return ''.join(
            [
                self.index_to_token.get(token_index, '')
                for token_index in token_indexes
                if token_index in self.sequence_tokens
            ]
        )
