import json
from typing import Dict

from importlib_resources import as_file, files

from .smiles_language import SELFIESLanguage, SMILESLanguage, SMILESTokenizer  # noqa

with as_file(
    files('pytoda.smiles.metadata').joinpath('tokenizer', 'vocab.json')
) as vocab_filepath:
    with open(vocab_filepath, 'r') as f:
        vocab: Dict[str, int] = json.load(f)
