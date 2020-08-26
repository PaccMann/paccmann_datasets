"""Processing utilities for .smi files."""
import pandas as pd
from rdkit import Chem
from ..files import read_smi


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
        pd.DataFrame([
            [row['SMILES'], index]
            for index, row in chunk.iterrows()
            if Chem.MolFromSmiles(row['SMILES'])
        ]).to_csv(
            output_filepath,
            index=None, header=None,
            mode='a',
            sep='\t'
        )
