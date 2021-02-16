import argparse
import os

from pytoda.smiles import SMILESLanguage

parser = argparse.ArgumentParser()
parser.add_argument(
    'smiles_language_filepath',
    type=str,
    help='path to a .pkl file of a legacy smiles language.',
)
parser.add_argument(
    'pretrained_path',
    type=str,
    help='path to a folder to store the language as text files.',
)


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    smiles_language = SMILESLanguage()
    smiles_language._from_legacy_pickled_language(args.smiles_language_filepath)

    # save tokenizer
    os.makedirs(args.pretrained_path)
    smiles_language.save_pretrained(args.pretrained_path)
