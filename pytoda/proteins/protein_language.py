"""Protein language handling."""
import logging
from typing import Iterator

import dill
from upfp import parse_fasta

from ..files import read_smi
from ..types import Indexes, Tokenizer, Tokens
from .processing import HUMAN_KINASE_ALIGNMENT_VOCAB, IUPAC_VOCAB, UNIREP_VOCAB

logger = logging.getLogger(__name__)


class ProteinLanguage(object):
    """
    ProteinLanguage class.

    ProteinLanguage handle Protein data defining the vocabulary and
    utilities to manipulate it.
    """

    unknown_token = '<UNK>'

    def __init__(
        self,
        name: str = 'protein-language',
        amino_acid_dict: str = 'iupac',
        tokenizer: Tokenizer = list,
        add_start_and_stop: bool = True,
    ) -> None:
        """
        Initialize Protein language.

        Args:
            name (str): name of the ProteinLanguage.
            amino_acid_dict (str): Tokenization regime for amino acid
                sequence. Defaults to 'iupac', alternative is 'unirep' or
                'human-kinase-alignment'.
            tokenizer (Tokenizer): This needs to be a function used to tokenize
                the amino acid sequences. The default is list which simply
                splits the sequence character-by-character.
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
        elif self.dict == 'human-kinase-alignment':
            self.token_to_index = HUMAN_KINASE_ALIGNMENT_VOCAB
        else:
            raise ValueError(
                "Choose dict as 'iupac' or 'unirep' or 'human-kinase-alignment' "
                f"(given was {amino_acid_dict})."
            )
        self.tokenizer = tokenizer
        self.setup_dict()

    def setup_dict(self) -> None:
        """
        Setup the dictionary.

        """
        # Setup dictionary
        self.sequence_tokens = [
            index for token, index in self.token_to_index.items() if '<' not in token
        ]
        self.number_of_tokens = len(self.token_to_index)
        self.index_to_token = {
            index: token for token, index in self.token_to_index.items()
        }

        if self.add_start_and_stop:
            self.max_token_sequence_length = 2
            self._get_total_number_of_tokens_fn = lambda tokens: len(tokens) + 2
            self._finalize_token_indexes_fn = lambda token_indexes: (
                [self.token_to_index['<START>']]
                + token_indexes
                + [self.token_to_index['<STOP>']]
            )
        else:
            self.max_token_sequence_length = 0
            self._get_total_number_of_tokens_fn = len
            self._finalize_token_indexes_fn = lambda token_indexes: token_indexes

        self.padding_index = self.token_to_index['<PAD>']
        self.start_index = self.token_to_index['<START>']
        self.stop_index = self.token_to_index['<STOP>']

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
        try:
            with open(filepath, 'rb') as f:
                protein_language = dill.load(f)
        except TypeError:
            # Necessary to load python3.7 pickled objects with >=3.8
            # For details see: https://github.com/uqfoundation/dill/pull/406
            storage = dill._dill._reverse_typemap['CodeType']
            dill._dill._reverse_typemap['CodeType'] = dill._dill._create_code
            with open(filepath, 'rb') as f:
                protein_language = dill.load(f)
            dill._dill._reverse_typemap['CodeType'] = storage
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

    def add_file(
        self,
        filepath: str,
        file_type: str = '.smi',
        index_col: int = 1,
        chunk_size: int = 100000,
    ) -> None:
        """
        Add a set of protein sequences from a file.

        Args:
            filepath (str): path to the file.
            file_type (str): Type of file, from {'.smi', '.csv', '.fasta',
                '.fasta.gz'}. If '.csv' is selected, it is assumed to be tab-
                separated.
            chunk_size (int): number of rows to read in a chunk.
                Defaults to 100000. Does not apply for fasta files.
            index_col (int): Data column used for indexing, defaults to 1, does
                not apply to fasta files.
        """
        if file_type not in ['.csv', '.smi', '.fasta', '.fasta.gz']:
            raise ValueError(
                "Please provide file of type {'.smi', '.csv', '.fasta','.fasta.gz'}"
            )

        if file_type == '.csv' or file_type == '.smi':
            try:
                for chunk in read_smi(
                    filepath,
                    chunk_size=chunk_size,
                    index_col=index_col,
                    names=['Sequence'],
                ):
                    for sequence in chunk['Sequence']:
                        self.add_sequence(sequence)
            except Exception:
                raise KeyError(
                    ".smi file needs to have 2 columns, index needs to be in "
                    f"column ({index_col}), sequences in the other."
                )
        elif file_type == '.fasta':
            database = parse_fasta(filepath, gzipped=False)
            for item in database:
                self.add_sequence(item['sequence'])

    def add_sequence(self, sequence: str) -> None:
        """
        Add a amino acid sequence to the language.

        Args:
            sequence (str): a sequence of amino acids.
        """
        tokens = self.tokenizer(sequence)
        self._update_max_token_sequence_length(tokens)

    def sequence_to_token_indexes_generator(self, sequence: str) -> Iterator[int]:
        """
        Transform tokens into indexes using a generator

        Args:
            sequence (str): an AAS representations

        Yields:
            Generator[int]: The generator of token indexes.
        """
        for token in self.tokenizer(sequence):
            if token not in self.token_to_index:
                logger.error(
                    'Replacing unknown token %s with %r', token, self.unknown_token
                )
                token = self.unknown_token
            yield self.token_to_index[token]

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
            list(self.sequence_to_token_indexes_generator(sequence))
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

    @property
    def method(self) -> str:
        """A string denoting the language method"""
        return self.dict
