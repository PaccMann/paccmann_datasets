"""Utilities for file handling."""
import os
import pandas as pd
from itertools import takewhile, repeat
from typing import Sequence


def count_file_lines(filepath: str, buffer_size: int = 1024 * 1024) -> int:
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
            lambda x: x, (raw_fp.read(buffer_size) for _ in repeat(None))
        ):
            number_of_lines += buffer.count(new_line)
            previous_buffer = buffer
        number_of_lines += int(not previous_buffer.endswith(new_line))
    return number_of_lines


def read_smi(
    filepath: str,
    chunk_size: int = None,
    index_col: int = 1,
    names: Sequence[str] = ['SMILES']
) -> pd.DataFrame:
    """
    Read a .smi (or .csv file with tab-separated values) in a pd.DataFrame.

    Args:
        filepath (str): path to a .smi file.
        chunk_size (int): size of the chunk. Defaults to None, a.k.a. no
            chunking.
        index_col (int): Data column used for indexing, defaults to 1.
        names (Sequence[str]): User-assigned names given to the columns.

    Returns:
        pd.DataFrame: a pd.DataFrame containing the data of the .smi file
            where the index is the index_col column.
    """

    return pd.read_csv(
        filepath,
        sep='\t',
        header=None,
        index_col=index_col,
        names=names,
        chunksize=chunk_size
    )
