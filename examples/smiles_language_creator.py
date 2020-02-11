#!/usr/bin/env python3
"""Create and export a SMILESLanguage given .smi files."""
import os
import argparse
from pytoda.smiles.smiles_language import SMILESLanguage

# define the parser arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    'smi_path', type=str, help='path to a folder with .smi files'
)
parser.add_argument(
    'output_filepath', type=str, help='path to a output .csv file'
)
add_start_and_stop_parser = parser.add_mutually_exclusive_group(required=False)
add_start_and_stop_parser.add_argument(
    '--add_start_and_stop', dest='add_start_and_stop', action='store_true'
)
add_start_and_stop_parser.add_argument(
    '--no_start_and_stop', dest='add_start_and_stop', action='store_false'
)
parser.set_defaults(add_start_and_stop=True)


def create_smiles_language(
    smi_path: str, output_filepath: str, add_start_and_stop: bool
) -> None:
    """
    Create a SMILESLanguage object and save it to disk.

    Args:
        smi_path (str): path to a folder containing .smi files.
        output_filepath (str): path where to store the SMILESLanguage.
        add_start_and_stop (bool): whether <START> and <END> tokens are used.
    """
    smiles_language = SMILESLanguage(add_start_and_stop=add_start_and_stop)
    smiles_language.add_smis(
        [
            os.path.join(smi_path, smi_filename)
            for smi_filename in os.listdir(smi_path)
            if smi_filename.endswith('.smi')
        ]
    )
    smiles_language.save(output_filepath)


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    # run the creation and export
    create_smiles_language(
        args.smi_path, args.output_filepath, args.add_start_and_stop
    )
