"""Utilities for file handling."""
import os
import pandas as pd
from itertools import takewhile, repeat


def count_file_lines(
    filepath: str, buffer_size: int = 1024*1024
) -> int:
    """
    Cound lines in a file without persisting it in memory.

    Args:
        filepath (str): path to the file.
        buffer_size (int): size of the buffer.

    Returns:
        int: Number of lines in the file.
    """
    number_of_lines = 0
    new_line = os.linesep.encode()
    with open(filepath, 'rb') as fp:
        raw_fp = fp.raw
        previous_buffer = None
        for buffer in takewhile(
            lambda x: x,
            (raw_fp.read(buffer_size) for _ in repeat(None))
        ):
            number_of_lines += buffer.count(new_line)
            previous_buffer = buffer
        number_of_lines += int(not previous_buffer.endswith(new_line))
    return number_of_lines


def read_smi(filepath: str, chunk_size: int = None) -> pd.DataFrame:
    """
    Read a .smi in a pd.DataFrame.

    Args:
        filepath (str): path to a .smi file.
        chunk_size (int): size of the chunk.
            Defaults to None, a.k.a. no chunking.

    Returns:
        pd.DataFrame: a pd.DataFrame containing the SMILES
            where the index is the compound name.
    """
    return pd.read_csv(
        filepath, sep='\t',
        header=None, index_col=1, names=['SMILES'],
        chunksize=chunk_size
    )
