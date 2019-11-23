"""SMILES processing utilities."""
import re
from ..types import Tokens

# tokenizer
SMILES_TOKENIZER = re.compile(
    r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|'
    r'-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
)
# handle "-" character
SMILES_NORMALIZER = re.compile(r'-(\w)')
# smiles normalization dictionary

# TODO: This does not seem right
SMILES_NORMALIZATION_DICTIONARY = str.maketrans(
    {
        'c': 'C',
        'n': 'N',
        'h': 'H',
        's': 'S',
        'o': 'O'
    }
)


def apply_normalization_dictionary(smiles: str) -> str:
    """
    Apply a SMILES normalization dictionary. If applied, the SMILES is
    not case-sensitive (blind to aromatic vs. aliphatic structures)

    Args:
        smiles (str): a SMILES representation.

    Returns:
        str: SMILES normalized using `SMILES_NORMALIZATION_DICTIONARY`.
    """
    return SMILES_NORMALIZER.sub(
        r'\1', smiles.translate(SMILES_NORMALIZATION_DICTIONARY)
    )


def tokenize_smiles(smiles: str, normalize=False) -> Tokens:
    """
    Tokenize SMILES after (optionally) normalizing it.

    Args:
        smiles (str): a SMILES representation.
        normalize (bool): whether normalization is done.

    Returns:
        Tokens: the tokenized SMILES after an optional normalization.
    """
    return [
        token for token in SMILES_TOKENIZER.
        split(apply_normalization_dictionary(smiles) if normalize else smiles)
        if token
    ]
