import argparse
from pytoda.smiles import SMILESLanguage

parser = argparse.ArgumentParser()
parser.add_argument(
    'smiles_language_filepath', type=str,
    help='path to a .pkl file of a legacy smiles language.'
)
parser.add_argument(
    'vocab_filepath', type=str, help='path to a output .json file'
)


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    smiles_language = SMILESLanguage()
    smiles_language._legacy_load_vocab_from_pickled_language(
        args.smiles_language_filepath, include_metadata=True
    )
    smiles_language.save_vocab(args.vocab_filepath, include_metadata=True)
