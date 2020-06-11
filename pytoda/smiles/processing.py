"""SMILES processing utilities."""
import logging
import re

from SmilesPE.pretokenizer import kmer_tokenizer

from ..types import Tokens

# tokenizer
SMILES_TOKENIZER = re.compile(
    r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|'
    r'-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
)
logger = logging.getLogger(__name__)


def tokenize_smiles(smiles: str, regexp=None) -> Tokens:
    """
    Tokenize a character-level SMILES string.

    Args:
        smiles (str): a SMILES representation.
        regexp (None, re.Pattern): optionally pass a regexp for the
            tokenization. If none is passed, SMILES_TOKENIZER is used.
    Returns:
        Tokens: the tokenized SMILES.
    """
    smiles_tokenizer = SMILES_TOKENIZER if regexp is None else regexp
    return [token for token in smiles_tokenizer.split(smiles) if token]


def kmer_smiles_tokenizer(
    smiles: str, k: int, stride: int = 1, *args, **kwargs
) -> Tokens:
    """K-Mer SMILES tokenization following SMILES PE (Li et al. 2020)

    Args:
        smiles (str): SMILES string to be tokenized.
        k (int): Positive integer denoting the tuple/k-gram lengths.
        stride (int, optional): Stride used for k-mer generation. Higher values
            result in less tokens. Defaults to 1 (densely overlapping).
        args (): Optional arguments for `kmer_tokenizer`.
        kwargs (): Optional keyword arguments for `kmer_tokenizer`.

    Returns:
        Tokens: Tokenized SMILES sequence (list of str).
    """

    return kmer_tokenizer(smiles, ngram=k, stride=stride, *args, **kwargs)


def tokenize_selfies(selfies: str) -> Tokens:
    """Tokenize SELFIES.

    NOTE: Code adapted from selfies package (`def selfies_to_hot`):
        https://github.com/aspuru-guzik-group/selfies

    Args:
        selfies (str): a SELFIES representation (character-level).

    Returns:
        Tokens: the tokenized SELFIES.
    """
    try:
        selfies = selfies.replace('.', '[.]')  # to allow parsing unbound atoms
        selfies_char_list_pre = selfies[1:-1].split('][')
        return [
            '[' + selfies_element + ']'
            for selfies_element in selfies_char_list_pre
        ]
    except Exception:
        logger.warning(f'Error in tokenizing {selfies}. Returning empty list.')
        return ['']
