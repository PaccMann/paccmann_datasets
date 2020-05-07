"""Polymer language handling."""
from typing import Iterable
from collections import Counter
from .smiles_language import SMILESLanguage
from ..types import Indexes, SMILESTokenizer
from .processing import tokenize_smiles, SMILES_TOKENIZER


class PolymerLanguage(SMILESLanguage):
    """
    PolymerLanguage class.

    PolymerLanguage is an extension of SMILESLanguage. A polymer language is
    usually shared across several SMILES datasets.
    """

    def __init__(
        self,
        entity_names: Iterable[str],
        name: str = 'polymer-language',
        smiles_tokenizer: SMILESTokenizer = (
            lambda smiles: tokenize_smiles(smiles, regexp=SMILES_TOKENIZER)
        )
    ) -> None:
        """
        Initialize Polymer language.

        Args:
            entity_names (Iterable[str]): A list of entity names that the
                polymer language can distinguish.
            name (str): name of the PolymerLanguage.
            smiles_tokenizer (SMILESTokenizer): SMILES tokenization function.
                Defaults to tokenize_smiles.
        """

        SMILESLanguage.__init__(self, add_start_and_stop=True)

        self.entities = list(map(lambda x: x.capitalize(), entity_names))
        # self.current_entity = self.entities[0]
        self.start_entity_tokens, self.stop_entity_tokens = (
            list(map(lambda x: '<' + x.upper() + '_' + s + '>', entity_names))
            for s in ['START', 'STOP']
        )

        self.name = name
        self.smiles_tokenizer = smiles_tokenizer

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
                self.start_entity_tokens + self.stop_entity_tokens +
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

    def update_entity(self, entity: str) -> None:
        """
        Update the current entity of the Polymer language object

        Args:
            entity (str): a chemical entity (e.g. 'Monomer').

        Returns:
            None
        """
        assert (
            entity.capitalize() in self.entities
        ), f'Unknown entity was given ({entity})'
        self.current_entity = entity.capitalize()

    def smiles_to_token_indexes(self, smiles: str) -> Indexes:
        """
        Transform character-level SMILES into a sequence of token indexes.

        Args:
            smiles (str): a SMILES (or SELFIES) representation.

        Returns:
            Indexes: indexes representation for the SMILES/SELFIES provided.
        """
        return [
            self.token_to_index['<' + self.current_entity.upper() + '_START>']
        ] + [
            self.token_to_index[token]
            for token in self.smiles_tokenizer(smiles)
            if token in self.token_to_index
        ] + [
            self.token_to_index['<' + self.current_entity.upper() + '_STOP>']
        ]

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
                if token_index > 3 + len(self.entities) * 2
            ]
        )
