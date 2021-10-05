import json
import os
from importlib_recources import files, as_file
from typing import Dict
from .smiles_language import SELFIESLanguage, SMILESLanguage, SMILESTokenizer  # noqa

with as_file(
    files('pytoda.smiles.metadata').joinpath('tokenizer', 'vocab.json')
) as vocab_filepath:
    with open(vocab_filepath, 'r') as f:
        vocab: Dict[str, int] = json.load(f)
