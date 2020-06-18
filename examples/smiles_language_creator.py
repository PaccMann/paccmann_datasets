#!/usr/bin/env python3
"""Export vocabulary of SMILESLanguage from .smi files in directory."""
import os
import argparse
from pytoda.smiles.smiles_language import SMILESLanguage

# define the parser arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    'smi_path', type=str, help='path to a folder with .smi files'
)
parser.add_argument(
    'vocab_filepath', type=str, help='path to a output .json file'
)


def create_smiles_language(
    smi_path: str, vocab_filepath: str
) -> None:
    """
    Create a SMILESLanguage object and save it to disk.

    Args:
        smi_path (str): path to a folder containing .smi files.
        vocab_filepath (str): path where to store the vocabulary json.
    """
    smiles_language = SMILESLanguage()
    smiles_language.add_smis(
        [
            os.path.join(smi_path, smi_filename)
            for smi_filename in os.listdir(smi_path)
            if smi_filename.endswith('.smi')
        ]
    )
    smiles_language.save_vocab(vocab_filepath, include_metadata=True)


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    # run the creation and export
    create_smiles_language(
        args.smi_path, args.vocab_filepath
    )
