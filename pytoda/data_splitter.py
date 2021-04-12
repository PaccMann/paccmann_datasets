"""Data splitting utilties."""
import hashlib
import os
import random
from math import ceil
from typing import Tuple

import numpy as np
import pandas as pd

from .types import Files


def csv_data_splitter(
    data_filepaths: Files,
    save_path: str,
    data_type: str,
    mode: str,
    seed: int = 42,
    test_fraction: float = 0.1,
    number_of_columns: int = 12,
    **kwargs,
) -> Tuple[str, str]:
    """
    Function for generic splitting into train and test data in csv format.
    This is an eager splitter trying to fit the entire dataset into memory.

    Args:
        data_filepaths (Files): a list of .csv files that contain the data.
        save_path (str): folder to store the training/testing dataset.
        data_type (str): data type (only used as prefix for the saved files).
        mode (str): mode to split data from: "random" and "file".
            - random: does a random split across all samples in all files.
            - file: randomly splits the files into training and testing.
        seed (int): random seed used for the split. Defaults to 42.
        test_fraction (float): portion of samples in testing data.
            Defaults to 0.1.
        number_of_columns (int): number of columns used to create the hash.
            Defaults to 12.
        kwargs (dict): additional parameters for pd.read_csv.
    Returns:
        Tuple[str, str]: a tuple pointing to the train and test files.
    """
    # preparation
    random.seed(seed)
    data_filepaths = (
        [data_filepaths] if isinstance(data_filepaths, str) else data_filepaths
    )
    file_suffix = '{}_{}_fraction_{}_id_{}_seed_{}.csv'
    hash_fn = hashlib.md5()
    if not ('index_col' in kwargs):
        kwargs['index_col'] = 0

    # helper function to produce unique hash. Base hash on df.to_string()
    # but only consider first number_of_columns columns (slow otherwise)
    def _hash_from_df_columns(df, number_of_columns):
        return df.reindex(sorted(df.columns), axis=1).to_string(
            columns=sorted(df.columns)[:number_of_columns]
        )

    # NOTE: if all *.csv files contain a single sample only
    # the splitting modes collapse
    # file splitting mode
    if mode == 'file':
        if len(data_filepaths) < 2:
            raise RuntimeError(
                'mode={} requires at least two input files.'.format(mode)
            )
        file_indexes = np.arange(len(data_filepaths))
        random.shuffle(file_indexes)
        # compile splits (ceil ensures test_df is not empty)
        test_df = pd.concat(
            [
                pd.read_csv(data_filepaths[index], **kwargs)
                for index in file_indexes[: ceil(test_fraction * len(data_filepaths))]
            ]
        )
        train_df = pd.concat(
            [
                pd.read_csv(data_filepaths[index], **kwargs)
                for index in file_indexes[ceil(test_fraction * len(data_filepaths)) :]
            ]
        )
    # random splitting mode:
    # build a joint df from all samples, then split
    elif mode == 'random':
        df = pd.concat(
            [pd.read_csv(data_path, **kwargs) for data_path in data_filepaths],
            sort=False,
        )
        sample_indexes = np.arange(df.shape[0])
        random.shuffle(sample_indexes)
        splitting_index = ceil(test_fraction * df.shape[0])
        test_df = df.iloc[sample_indexes[:splitting_index]]
        train_df = df.iloc[sample_indexes[splitting_index:]]
    else:
        raise ValueError('Choose mode to be from the set {"random", "file"}.')
    # generate hash (per default uses first number_of_columns
    # columns to define hash)
    number_of_columns = (
        number_of_columns if test_df.shape[1] > number_of_columns else test_df.shape[1]
    )
    hash_str = _hash_from_df_columns(
        train_df, number_of_columns
    ) + _hash_from_df_columns(test_df, number_of_columns)
    hash_fn.update(hash_str.encode('utf-8'))
    hash_id = str(int(hash_fn.hexdigest(), 16))[:6]
    # generate file path
    train_filepath = os.path.join(
        save_path,
        file_suffix.format(data_type, 'train', 1 - test_fraction, hash_id, seed),
    )
    test_filepath = os.path.join(
        save_path, file_suffix.format(data_type, 'test', test_fraction, hash_id, seed)
    )
    # write the dataset
    for path, df in zip([train_filepath, test_filepath], [train_df, test_df]):
        df.to_csv(path)
    return train_filepath, test_filepath
