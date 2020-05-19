"""Amino Acid Sequence transforms."""
from ..transforms import Transform
from ..types import Indexes
from .protein_language import ProteinLanguage


class SequenceToTokenIndexes(Transform):
    """Transform Sequence to token indexes using Sequence language."""

    def __init__(self, protein_language: ProteinLanguage) -> None:
        """
        Initialize a Sequence to token indexes object.

        Args:
            protein_language (ProteinLanguage): a Protein language.
        """
        self.protein_language = protein_language

    def __call__(self, smiles: str) -> Indexes:
        """
        Apply the Sequence tokenization transformation

        Args:
            smiles (str): a Sequence representation.

        Returns:
            Indexes: indexes representation for the Sequence provided.
        """
        return self.protein_language.sequence_to_token_indexes(smiles)
