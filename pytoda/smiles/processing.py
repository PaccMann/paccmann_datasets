"""SMILES processing utilities."""
import codecs
import logging
import re
import warnings

from importlib_resources import as_file, files
from selfies import split_selfies as split_selfies_
from SmilesPE.pretokenizer import kmer_tokenizer
from SmilesPE.tokenizer import SPE_Tokenizer

from ..types import Dict, Tokenizer, Tokens

logger = logging.getLogger(__name__)

# tokenizer
SMILES_TOKENIZER = re.compile(
    r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|'
    r'-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
)

with as_file(files('pytoda.smiles.metadata').joinpath('spe_chembl.txt')) as filepath:
    SPE_TOKENIZER = SPE_Tokenizer(codecs.open(str(filepath)))


def tokenize_smiles(smiles: str, regexp=SMILES_TOKENIZER, *args, **kwargs) -> Tokens:
    """
    Tokenize a character-level SMILES string.

    Args:
        smiles (str): a SMILES representation.
        regexp (re.Pattern): optionally pass a regexp for the
            tokenization. Defaults to SMILES_TOKENIZER.
        args (): ignored, for backwards compatibility.
        kwargs (): ignored, for backwards compatibility.
    Returns:
        Tokens: the tokenized SMILES.
    """
    return [token for token in regexp.split(smiles) if token]


def kmer_smiles_tokenizer(
    smiles: str, k: int = 2, stride: int = 1, *args, **kwargs
) -> Tokens:
    """K-Mer SMILES tokenization following SMILES PE (Li et al. 2020):
        Li, Xinhao, and Denis Fourches. "SMILES Pair Encoding: A Data-Driven
        Substructure Tokenization Algorithm for Deep Learning." (2020).


    Args:
        smiles (str): SMILES string to be tokenized.
        k (int): Positive integer denoting the tuple/k-gram lengths. Defaults
            to 2 (bigrams).
        stride (int, optional): Stride used for k-mer generation. Higher values
            result in less tokens. Defaults to 1 (densely overlapping).
        args (): Optional arguments for `kmer_tokenizer`.
        kwargs (): Optional keyword arguments for `kmer_tokenizer`.

    Returns:
        Tokens: Tokenized SMILES sequence (list of str).
    """

    return kmer_tokenizer(smiles, ngram=k, stride=stride, *args, **kwargs)


def spe_smiles_tokenizer(smiles: str) -> Tokens:
    """Pretrained SMILES Pair Encoding tokenizer following (Li et al. 2020).
        Splits a SMILES into tokens of substructures of varying lengths,
        depending on occurrence of tokens in ChEMBL dataset.

        Li, Xinhao, and Denis Fourches. "SMILES Pair Encoding: A Data-Driven
        Substructure Tokenization Algorithm for Deep Learning." (2020).

    Args:
        smiles (str): SMILES string to be tokenized.

    Returns:
        Tokens: SMILES tokenized into substructures (list of str).
    """

    return SPE_TOKENIZER.tokenize(smiles).split(' ')


def tokenize_selfies(selfies: str) -> Tokens:
    """Tokenize SELFIES.

    NOTE: Code adapted from selfies package (`def selfies_to_hot`):
        https://github.com/aspuru-guzik-group/selfies

    Args:
        selfies (str): a SELFIES representation (character-level).

    Returns:
        Tokens: the tokenized SELFIES.
    """
    warnings.warn(
        "tokenize_selfies will be deprecated in favor of `split_selfies`",
        DeprecationWarning,
    )
    try:
        selfies = selfies.replace('.', '[.]')  # to allow parsing unbound atoms
        selfies_char_list_pre = selfies[1:-1].split('][')
        return [
            '[' + selfies_element + ']' for selfies_element in selfies_char_list_pre
        ]
    except Exception:
        logger.warning(f'Error in tokenizing {selfies}. Returning empty list.')
        return ['']


def split_selfies(selfies: str) -> Tokens:
    """Tokenize SELFIES, wrapping generator as list.

    Args:
        selfies (str): a SELFIES representation (character-level).

    Returns:
        Tokens: the tokenized SELFIES.
    """
    return list(split_selfies_(selfies))


TOKENIZER_FUNCTIONS: Dict[str, Tokenizer] = {
    'smiles': tokenize_smiles,
    'kmer_smiles': kmer_smiles_tokenizer,
    'spe_smiles': spe_smiles_tokenizer,
    'selfies': split_selfies,
}
