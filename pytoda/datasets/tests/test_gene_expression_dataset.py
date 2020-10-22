"""Testing GeneExpressionDataset with eager backend."""
import unittest
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pytoda.datasets import GeneExpressionDataset
from pytoda.tests.utils import TestFileContent

CONTENT = os.linesep.join(
    [
        'genes,A,B,C,D',
        'sample_3,9.45,4.984,7.016,8.336',
        'sample_2,7.188,0.695,10.34,6.047',
        'sample_1,9.25,6.133,5.047,5.6',
    ]
)
MORE_CONTENT = os.linesep.join(
    [
        'genes,B,C,D,E,F',
        'sample_10,4.918,0.0794,1.605,3.463,10.18',
        'sample_11,3.043,8.56,1.961,0.6226,5.027',
        'sample_12,4.76,1.124,6.06,0.3743,11.05',
        'sample_13,0.626,5.164,4.277,4.414,2.7',
    ]
)


class TestGeneExpressionDatasetEagerBackend(unittest.TestCase):
    """Testing GeneExpressionDataset with eager backend."""

    def setUp(self):
        self.backend = 'eager'
        print(f'backend is {self.backend}')
        self.content = CONTENT
        self.other_content = MORE_CONTENT

    def test___len__(self) -> None:
        """Test __len__."""

        with TestFileContent(self.content) as a_test_file:
            with TestFileContent(self.other_content) as another_test_file:
                gene_expression_dataset = GeneExpressionDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    backend=self.backend,
                    index_col=0
                )
                self.assertEqual(len(gene_expression_dataset), 7)

    def test___getitem__(self) -> None:
        """Test __getitem__."""
        with TestFileContent(self.content) as a_test_file:
            with TestFileContent(self.other_content) as another_test_file:
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
                    backend=self.backend,
                    index_col=0
                )
                gene_list = gene_expression_dataset.gene_list
                mean = df.mean()[gene_list].values
                std = df.std(ddof=0)[gene_list].values
                for i, (key, row) in enumerate(df[gene_list].iterrows()):
                    np.testing.assert_almost_equal(
                        gene_expression_dataset[i].numpy(),
                        (row.values - mean) / std, 5
                    )
                np.testing.assert_almost_equal(
                    gene_expression_dataset.mean, mean, 5
                )
                np.testing.assert_almost_equal(
                    gene_expression_dataset.std, std, 5
                )

                gene_expression_dataset = GeneExpressionDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    backend=self.backend,
                    index_col=0,
                    standardize=False,
                    min_max=True,
                )
                minimum = df.min()[gene_list].values
                maximum = df.max()[gene_list].values
                diff = maximum - minimum
                for i, (key, row) in enumerate(df[gene_list].iterrows()):
                    np.testing.assert_almost_equal(
                        gene_expression_dataset[i].numpy(),
                        (row.values - minimum) / diff, 5
                    )
                np.testing.assert_almost_equal(
                    gene_expression_dataset.min, minimum, 5
                )
                np.testing.assert_almost_equal(
                    gene_expression_dataset.max, maximum, 5
                )

    def test_processing_parameters(self) -> None:
        with TestFileContent(self.content) as a_test_file, \
            TestFileContent(self.other_content) as another_test_file \
        :
            standard_dataset = GeneExpressionDataset(
                a_test_file.filename,
                another_test_file.filename,
                backend=self.backend,
                index_col=0
            )
            minmax_dataset = GeneExpressionDataset(
                a_test_file.filename,
                another_test_file.filename,
                backend=self.backend,
                index_col=0,
                standardize=False,
                min_max=True,
            )
            gene_list = standard_dataset.gene_list
            self.assertTrue(gene_list == minmax_dataset.gene_list)
            df = pd.concat(
                [
                    pd.read_csv(a_test_file.filename, index_col=0),
                    pd.read_csv(another_test_file.filename, index_col=0),
                ],
                sort=False
            )[gene_list]

            # flat = df.values.flatten()
            mean_array = df.mean().values
            std_array = df.std(ddof=0).values

            # with min max scaling we can check for values 0 and 1
            maximum_array = df.max().values
            minimum_array = df.min().values

            # scalar scaling (single max and min)
            max_n, max_p = map(
                int, np.unravel_index(np.argmax(df.values), df.shape)
            )
            min_n, min_p = map(
                int, np.unravel_index(np.argmin(df.values), df.shape)
            )
            for maximum, minimum in [
                # scalar
                [np.max(maximum_array),
                 np.min(minimum_array)],
                # list length 1
                [[np.max(maximum_array)], [np.min(minimum_array)]],
            ]:
                processing_parameters = {
                    'max': maximum,
                    'min': minimum,
                }
                minmax_ds = GeneExpressionDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    backend=self.backend,
                    index_col=0,
                    standardize=False,
                    min_max=True,
                    processing_parameters=processing_parameters,
                )

                self.assertEqual(minmax_ds[max_n][max_p], 1)
                self.assertEqual(minmax_ds[min_n][min_p], 0)

            # array scaling (feature wise max and min)
            max_indeces = map(int, np.argmax(df.values, axis=0))
            min_indeces = map(int, np.argmin(df.values, axis=0))

            for maximum, minimum in [
                # list
                [maximum_array.tolist(),
                 minimum_array.tolist()],
                # ndarray
                [maximum_array, minimum_array]
            ]:
                processing_parameters = {
                    'max': maximum,
                    'min': minimum,
                }
                minmax_ds = GeneExpressionDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    backend=self.backend,
                    index_col=0,
                    standardize=False,
                    min_max=True,
                    processing_parameters=processing_parameters,
                )
                # check max_index / min_index are 1 and 0 per feature
                for feature_index, sample_index in enumerate(max_indeces):
                    self.assertEqual(minmax_ds[sample_index][feature_index], 1)
                for feature_index, sample_index in enumerate(min_indeces):
                    self.assertEqual(minmax_ds[sample_index][feature_index], 0)

    def test_data_loader(self) -> None:
        """Test data_loader."""
        gene_subset_list = ['B', 'D', 'F']
        with TestFileContent(self.content) as a_test_file:
            with TestFileContent(self.other_content) as another_test_file:
                gene_expression_dataset = GeneExpressionDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    gene_list=gene_subset_list,
                    backend=self.backend,
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

    def _test_indexed(self, ds, keys, index):
        key = keys[index]
        positive_index = index % len(ds)
        # get_key (support for negative index?)
        self.assertEqual(key, ds.get_key(positive_index))
        self.assertEqual(key, ds.get_key(index))
        # get_index
        self.assertEqual(positive_index, ds.get_index(key))
        # get_item_from_key
        self.assertTrue(all(ds[index] == ds.get_item_from_key(key)))
        # keys
        self.assertSequenceEqual(keys, list(ds.keys()))
        # duplicate keys
        self.assertFalse(ds.has_duplicate_keys)

    def test_all_base_for_indexed_methods(self):

        with TestFileContent(self.content) as a_test_file:
            with TestFileContent(self.other_content) as another_test_file:
                gene_expression_dataset = GeneExpressionDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    backend=self.backend,
                    index_col=0
                )
                gene_expression_ds_0 = GeneExpressionDataset(
                    a_test_file.filename, backend=self.backend, index_col=0
                )
                gene_expression_ds_1 = GeneExpressionDataset(
                    another_test_file.filename,
                    backend=self.backend,
                    index_col=0
                )
        all_keys = [
            row.split(',')[0] for row in self.content.split('\n')[1:] +
            self.other_content.split('\n')[1:]
        ]

        for ds, keys in [
            (gene_expression_dataset, all_keys),
            (gene_expression_ds_0, all_keys[:len(gene_expression_ds_0)]),
            (gene_expression_ds_1, all_keys[len(gene_expression_ds_0):]),
            (gene_expression_ds_0 + gene_expression_ds_1, all_keys),
        ]:
            index = -1
            self._test_indexed(ds, keys, index)

        # duplicate
        duplicate_ds = gene_expression_ds_0 + gene_expression_ds_0
        self.assertTrue(duplicate_ds.has_duplicate_keys)

        # GeneExpressionDataset does not test and raise
        with TestFileContent(self.content) as a_test_file:
            gene_expression_dataset = GeneExpressionDataset(
                a_test_file.filename,
                a_test_file.filename,
                backend=self.backend,
                index_col=0
            )
            self.assertTrue(gene_expression_dataset.has_duplicate_keys)


class TestGeneExpressionDatasetLazyBackend(
    TestGeneExpressionDatasetEagerBackend
):  # noqa
    """Testing GeneExpressionDataset with lazy backend."""

    def setUp(self):
        self.backend = 'lazy'
        print(f'backend is {self.backend}')
        self.content = CONTENT
        self.other_content = MORE_CONTENT


if __name__ == '__main__':
    unittest.main()
