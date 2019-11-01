#!/usr/bin/env python3
"""Split in train and test a .csv."""
import sys
import logging
import argparse
from pytoda.data_splitter import csv_data_splitter

# setting up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger('split_csv_data')

# define the parser arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '-f',
    '--filepaths',
    nargs='+',
    help='list of files to use to generate the splits',
    required=True
)
parser.add_argument(
    '-o',
    '--output_path',
    type=str,
    help='output path where to store the splits',
    required=True
)
parser.add_argument(
    '-d',
    '--data_type',
    type=str,
    help='data type, used to generate the output file names',
    required=True
)
parser.add_argument(
    '-m',
    '--mode',
    type=str,
    help='strategy used to split the data',
    choices=['random', 'file'],
    required=True
)
parser.add_argument(
    '-s',
    '--seed',
    type=int,
    help=('seed used by the random generator. '
          'Defaults to 42'),
    default=42
)
parser.add_argument(
    '-t',
    '--test_fraction',
    type=float,
    help=('portion of samples in testing data. '
          'Defaults to 0.1'),
    default=0.1
)
parser.add_argument(
    '-n',
    '--number_of_columns',
    type=int,
    help=('number of columns used to generate a hash. '
          'Defaults to 12'),
    default=12
)
parser.add_argument(
    '-i',
    '--index_col',
    type=int,
    help=('index column in the .csv flies. '
          'Defaults to 0'),
    default=0
)
parser.add_argument(
    '--separator',
    type=str,
    help=('separators in the .csv files. '
          'Defaults to ","'),
    default=','
)
parser.add_argument(
    '--header',
    type=int,
    help=('header row in the .csv files. '
          'Defaults to 0'),
    default=0
)

if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    # run the split
    train_filepath, test_filepath = csv_data_splitter(
        data_filepaths=args.filepaths,
        save_path=args.output_path,
        data_type=args.data_type,
        mode=args.mode,
        seed=args.seed,
        test_fraction=args.test_fraction,
        number_of_columns=args.number_of_columns,
        index_col=args.index_col,
        sep=args.separator,
        header=args.header
    )
    logger.info(
        'Data splitted into train_filepath={} and test_filepath={}.'.format(
            train_filepath, test_filepath
        )
    )
