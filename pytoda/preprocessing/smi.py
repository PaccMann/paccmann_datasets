"""Processing utilities for .smi files."""
import os
import sys
from typing import List

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from ..files import read_smi
from ..smiles.transforms import Canonicalization


def filter_invalid_smi(
    input_filepath: str, output_filepath: str, chunk_size: int = 100000
):
    """
    Execute chunked invalid SMILES filtering in a .smi file.

    Args:
        input_filepath (str): path to the .smi file to process.
        output_filepath (str): path where to store the filtered .smi file.
        chunk_size (int): size of the SMILES chunk. Defaults to 100000.
    """
    for chunk in read_smi(input_filepath, chunk_size=chunk_size):
        pd.DataFrame(
            [
                [row['SMILES'], index]
                for index, row in chunk.iterrows()
                if Chem.MolFromSmiles(row['SMILES'])
            ]
        ).to_csv(output_filepath, index=None, header=None, mode='a', sep='\t')


def find_undesired_smiles_files(
    undesired_filepath: str,
    data_filepath: str,
    save_matches: bool = False,
    file=sys.stdout,
    **smi_kwargs,
):
    """
    Method to find undesired SMILES in a list of existing SMILES.

    Args:
        undesired_filepath (str): Path to .smi file with a header at first row.
        data_filepath (str): Path to .csv file with a column 'SMILES'.
        save_matches (bool, optional): Whether found matches should be plotted and
            saved. Defaults to False.
    """
    canonicalizer = Canonicalization()

    # Read undesired data
    undesired = read_smi(undesired_filepath, **smi_kwargs)
    undesired_smiles = undesired['SMILES'].apply(canonicalizer).tolist()

    # Read data filepath
    df = pd.read_csv(data_filepath)

    matches, idxs = [], []
    for idx, row in df.iterrows():
        match = find_undesired_smiles(row['SMILES'], undesired_smiles, canonical=True)

        if match:
            print(f'Found {row.SMILES} in list of undesired SMILES.', file=file)
            matches.append(row.SMILES)
            idxs.append(idx)

    if len(matches) == 0:
        print('No matches found, shutting down.', file=file)
        return

    if save_matches:
        grid = Draw.MolsToGridImage(
            [Chem.MolFromSmiles(s) for s in matches],
            molsPerRow=5,
            maxMols=50,
            legends=[f'Idx: {i}, {s}' for i, s in zip(idxs, matches)],
        )
        grid.save(
            os.path.join(os.path.dirname(data_filepath), 'undesired_molecules.pdf')
        )

    return


def find_undesired_smiles(
    smiles: str, undesired_smiles: List, canonical: bool = False
) -> bool:
    """
    Whether or not a given SMILES is contained in a list of SMILES, respecting
    canonicalization.

    Args:
        smiles (str): Seed SMILES.
        undesired_smiles (List): List of SMILES for comparison
        canonical (bool, optional): Whether comparison list was canonicalized.
            Defaults to False.

    Returns:
        bool: Whether SMILES was present in undesired_smiles.
    """

    canonicalizer = Canonicalization()
    if not canonical:
        undesired_smiles = list(map(canonicalizer, undesired_smiles))

    smiles = canonicalizer(smiles)

    return smiles in undesired_smiles
