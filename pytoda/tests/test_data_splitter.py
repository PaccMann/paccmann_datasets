"""Testing data splitter."""
import os
import tempfile
import unittest
from typing import Tuple

import pandas as pd

from pytoda.data_splitter import csv_data_splitter
from pytoda.tests.utils import TestFileContent
from pytoda.types import Files


class TestDataSplitter(unittest.TestCase):
    """Testing csv data splitting."""

    def _read_dfs(self, filepaths: Files) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Read data frames from a list of files.

        Args:
            filepaths (Files): a list of files.

        Returns:
            Tuple[pd.DataFrame, ...]: a tuple of data frames.
        """
        return tuple(pd.read_csv(filepath, index_col=0) for filepath in filepaths)

    def test_data_splitter(self) -> None:
        """Test csv_data_splitter."""

        a_content = os.linesep.join(
            [
                'genes,A,B,C,D',
                'sample_3,9.45,4.984,7.016,8.336',
                'sample_2,7.188,0.695,10.34,6.047',
                'sample_1,9.25,6.133,5.047,5.6',
            ]
        )
        another_content = os.linesep.join(
            [
                'genes,B,C,D,E,F',
                'sample_10,4.918,0.0794,1.605,3.463,10.18',
                'sample_11,3.043,8.56,1.961,0.6226,5.027',
                'sample_12,4.76,1.124,6.06,0.3743,11.05',
                'sample_13,0.626,5.164,4.277,4.414,2.7',
            ]
        )

        # first row for random splits, second for file split
        ground_truth = [
            [(6, 6), (1, 6)],
            [(5, 6), (2, 6)],
            [(3, 6), (4, 6)],
            [(3, 4), (4, 5)],
            [(3, 4), (4, 5)],
            [(3, 4), (4, 5)],
        ]
        index = 0
        with tempfile.TemporaryDirectory() as directory:
            for mode in ['random', 'file']:
                for test_fraction in [0.1, 0.2, 0.5]:
                    with TestFileContent(a_content) as a_test_file:
                        with TestFileContent(another_content) as another_test_file:

                            train_filepath, test_filepath = csv_data_splitter(
                                [a_test_file.filename, another_test_file.filename],
                                directory,
                                'general',
                                mode=mode,
                                test_fraction=test_fraction,
                            )
                            train_df, test_df = self._read_dfs(
                                [train_filepath, test_filepath]
                            )
                            self.assertEqual(
                                ground_truth[index], [train_df.shape, test_df.shape]
                            )
                            index += 1


if __name__ == '__main__':
    unittest.main()
