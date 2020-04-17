"""Protein language handling."""

import dill
from upfp import parse_fasta

from ..files import read_smi
from ..types import Indexes, Tokens
from .processing import AA_PROPERTIES_NUM, AA_FEAT, BLOSUM62
from .protein_language import ProteinLanguage

class ProteinFeatureLanguage(ProteinLanguage):
    """
    ProteinFeatureLanguage class.

    ProteinFeatureLanguage handles Protein data and translates from text to 
    feature space
    """

    def __init__(
        self,
        name: str = 'protein-feature-language',
        features: str = 'blosum',
        tokenizer: object = list,
        add_start_and_stop: bool = True
    ) -> None:

        """
        Initialize Protein feature language.

        Args:
            name (str): name of the ProteinFeatureLanguage.
            features (str): Feature alphabet choice. Defaults to 'blosum', 
            alternatives are 'binary_features' and 'float_features'.
            tokenizer (object): This needs to be a function used to tokenize
                the amino acid sequences. The default is list which simply
                splits the sequence character-by-character.
        """
        self.name = name
        self.feat = features
        self.add_start_and_stop = add_start_and_stop

        if self.feat == 'binary_features':
        	self.token_to_index = AA_PROPERTIES_NUM
        elif self.feat == 'float_features':
        	self.token_to_index = AA_FEAT
        elif self.feat == 'blosum':
        	self.token_to_index = BLOSUM62
        else:
            raise ValueError(
                "Choose dict as 'binary_features', 'float_features' or 'blosum' (given was"
                f"{amino_acid_dict})."
            )
        # Setup dictionary
        self.sequence_tokens = [
            index for token, index in self.token_to_index.items()
            if '<' not in token
        ]

        self.tokenizer = tokenizer
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

        self.padding_index = self.token_to_index['<PAD>']
        self.start_index = self.token_to_index['<START>']
        self.stop_index = self.token_to_index['<STOP>']


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
