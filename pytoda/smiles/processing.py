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


def tokenize_smiles(smiles: str, normalize=False) -> Tokens:
    """
    Tokenize a character-level SMILES string.

    Args:
        smiles (str): a SMILES representation.
        normalize (bool): whether normalization is done.
        
        NOTE: The `normalize` argument is deprecated and will be removed in a
        future release.

    Returns:
        Tokens: the tokenized SMILES.
    """
    return [token for token in SMILES_TOKENIZER.split(smiles) if token]


def tokenize_selfies(selfies: str) -> Tokens:
    """Tokenize SELFIES.

    NOTE: Code adapted from selfies package (`def selfies_to_hot`):
        https://github.com/aspuru-guzik-group/selfies

    Args:
        selfies (str): a SELFIES representation (character-level).

    Returns:
        Tokens: the tokenized SELFIES.
    """

    selfies = selfies.replace('.', '[.]')  # to allow parsing unbound atoms
    selfies_char_list_pre = selfies[1:-1].split('][')
    return [
        '[' + selfies_element + ']'
        for selfies_element in selfies_char_list_pre
    ]
