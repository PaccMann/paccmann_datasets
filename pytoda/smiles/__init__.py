import json
import os

from .smiles_language import SELFIESLanguage, SMILESLanguage, SMILESTokenizer  # noqa

with open(
    os.path.join(os.path.dirname(__file__), 'metadata', 'tokenizer', 'vocab.json'), 'r'
) as f:
    vocab = json.load(f)
