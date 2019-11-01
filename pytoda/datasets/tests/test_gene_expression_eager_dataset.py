"""Testing GeneExpressionDataset with eager backend."""
import unittest
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pytoda.datasets import GeneExpressionDataset
from pytoda.tests.utils import TestFileContent


class TestGeneExpressionDatasetEagerBackend(unittest.TestCase):
    """Testing GeneExpressionDataset with eager backend."""

    def test___len__(self) -> None:
        """Test __len__."""
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
        with TestFileContent(a_content) as a_test_file:
            with TestFileContent(another_content) as another_test_file:
                gene_expression_dataset = GeneExpressionDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    backend='eager',
                    index_col=0
                )
                self.assertEqual(len(gene_expression_dataset), 7)

    def test___getitem__(self) -> None:
        """Test __getitem__."""
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
        with TestFileContent(a_content) as a_test_file:
            with TestFileContent(another_content) as another_test_file:
                df = pd.concat(
                    [
                        pd.read_csv(a_test_file.filename, index_col=0),
                        pd.read_csv(another_test_file.filename, index_col=0),
                    ],
                    sort=False
                )
                gene_expression_dataset = GeneExpressionDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    backend='eager',
                    index_col=0
                )
                gene_list = gene_expression_dataset.gene_list
                mean = df.mean()[gene_list].values
                std = df.std(ddof=0)[gene_list].values
                np.testing.assert_almost_equal(
                    gene_expression_dataset[4].numpy(),
                    (df[gene_list].iloc[4].values - mean) / std, 5
                )
                np.testing.assert_almost_equal(
                    gene_expression_dataset.mean, mean, 5
                )
                np.testing.assert_almost_equal(
                    gene_expression_dataset.std, std, 5
                )

    def test_data_loader(self) -> None:
        """Test data_loader."""
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
        gene_subset_list = ['B', 'D', 'F']
        with TestFileContent(a_content) as a_test_file:
            with TestFileContent(another_content) as another_test_file:
                gene_expression_dataset = GeneExpressionDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    gene_list=gene_subset_list,
                    backend='eager',
                    index_col=0
                )
                data_loader = DataLoader(
                    gene_expression_dataset, batch_size=2, shuffle=True
                )
                for batch_index, batch in enumerate(data_loader):
                    self.assertEqual(
                        batch.shape, (
                            1 if batch_index == 3 else 2,
                            gene_expression_dataset.number_of_features
                        )
                    )
                    if batch_index > 2:
                        break


if __name__ == '__main__':
    unittest.main()
