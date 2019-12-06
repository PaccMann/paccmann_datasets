#!/usr/bin/env python3
"""Convert SMILES to Morgan fingerprints."""
import argparse
import pandas as pd
from pytoda.files import read_smi
from pytoda.smiles.transforms import SMILESToMorganFingerprints

# define the parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('smi_filepath', type=str, help='path to a input .smi file')
parser.add_argument(
    'output_filepath', type=str, help='path to a output .csv file'
)
parser.add_argument(
    '-r',
    '--radius',
    type=int,
    help=('radius for the fingerprints. '
          'Defaults to 2'),
    default=2
)
parser.add_argument(
    '-b',
    '--bits',
    type=int,
    help=('number of bits composing the fingerprint. '
          'Defaults to 512'),
    default=512
)


def convert_smi_to_fingerprints(
    smi_filepath: str, output_filepath: str, radius: int, bits: int,
    chirality: bool
) -> None:
    """
    Convert a .smi in .csv containing Morgan fingerprints.

    Args:
        smi_filepath (str): path to a input .smi file.
        output_filepath (str): path to a output .csv file.
        radius (int): radius for the fingerprints.
        bits (int): number of bits composing the fingerprint.
    """
    smiles_df = read_smi(smi_filepath)
    converter = SMILESToMorganFingerprints(
        radius=radius, bits=bits, chirality=chirality
    )
    fingerprints_df = pd.DataFrame(
        [converter(smiles) for smiles in smiles_df['SMILES']],
        index=smiles_df.index,
        columns=list(range(bits))
    )
    fingerprints_df.to_csv(output_filepath)


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    # run the conversion
    convert_smi_to_fingerprints(
        args.smi_filepath, args.output_filepath, args.radius, args.bits
    )
