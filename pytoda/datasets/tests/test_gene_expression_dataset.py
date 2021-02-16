"""Testing GeneExpressionDataset with eager backend."""
import os
import unittest

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from pytoda.datasets import GeneExpressionDataset
from pytoda.tests.utils import TestFileContent

CONTENT = os.linesep.join(
    [
        'genes,A,C,B,D',
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
                    index_col=0,
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
                    sort=False,
                )
                gene_expression_dataset = GeneExpressionDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    backend=self.backend,
                    index_col=0,
                )
                gene_list = gene_expression_dataset.gene_list
                mean = df.mean()[gene_list].values
                std = df.std(ddof=0)[gene_list].values
                for i, (key, row) in enumerate(df[gene_list].iterrows()):
                    np.testing.assert_almost_equal(
                        gene_expression_dataset[i].numpy(), (row.values - mean) / std, 5
                    )
                np.testing.assert_almost_equal(gene_expression_dataset.mean, mean, 5)
                np.testing.assert_almost_equal(gene_expression_dataset.std, std, 5)

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
                        (row.values - minimum) / diff,
                        5,
                    )
                np.testing.assert_almost_equal(gene_expression_dataset.min, minimum, 5)
                np.testing.assert_almost_equal(gene_expression_dataset.max, maximum, 5)

    def test_processing_parameters_standardize_reindex(self) -> None:
        with TestFileContent(self.content) as a_test_file, TestFileContent(
            self.other_content
        ) as another_test_file:
            # feature not in data is filled with zeros
            feature_list = ['E', 'C', 'D', 'B', 'all_missing']
            standard_dataset = GeneExpressionDataset(
                a_test_file.filename,
                another_test_file.filename,
                gene_list=feature_list,
                backend=self.backend,
                index_col=0,
            )
            self.assertEqual(standard_dataset[0][-1], 0)

            gene_list = standard_dataset.gene_list
            df = pd.concat(
                [
                    pd.read_csv(a_test_file.filename, index_col=0),
                    pd.read_csv(another_test_file.filename, index_col=0),
                ],
                sort=False,
            ).reindex(
                columns=gene_list
            )  # , fill_value=0.0)

            # scalar scaling (single max and min)
            flat = df.values.flatten()
            # allow nan
            mean_float = np.nanmean(flat)
            std_float = np.nanstd(flat)
            for mean, std in [
                # scalar
                [mean_float, std_float],
                # list length 1
                [[mean_float], [std_float]],
            ]:
                processing_parameters = {
                    'mean': mean,
                    'std': std,
                }
                standard_ds = GeneExpressionDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    gene_list=feature_list,
                    backend=self.backend,
                    index_col=0,
                    standardize=True,
                    min_max=False,
                    processing_parameters=processing_parameters,
                    impute=None,
                )

                # collect flat values
                ds_1d = np.concatenate([item.numpy() for item in standard_ds])
                # allowing/ignoring nan
                np.testing.assert_almost_equal(np.nanmean(ds_1d), 0)
                np.testing.assert_almost_equal(np.nanstd(ds_1d), 1)

            mean_array = df.mean().values
            std_array = df.std(ddof=0).values
            # NOTE: numpy and pytoda use ddof of 0, whereas pandas default is 1
            for mean, std in [
                # list
                [mean_array.tolist(), std_array.tolist()],
                # ndarray
                [mean_array, std_array],
            ]:
                processing_parameters = {
                    'mean': mean,
                    'std': std,
                }
                standard_ds = GeneExpressionDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    gene_list=feature_list,
                    backend=self.backend,
                    index_col=0,
                    standardize=True,
                    min_max=False,
                    processing_parameters=processing_parameters,
                    impute=None,
                )

                # collect transformed values
                ds_2d = np.stack([item.numpy() for item in standard_ds])
                # contains nan
                ds_means = np.nanmean(ds_2d, axis=0)
                ds_stds = np.nanstd(ds_2d, axis=0)

                # debug std
                # standard_ds.std
                # std_array        # == df.std(ddof=0).values
                # df.std().values  # == df.std(ddof=1).values

                for index, feature in enumerate(standard_ds.gene_list):
                    if feature == 'all_missing':
                        # no features at all so transformed stat is nan
                        self.assertTrue(np.isnan(ds_means[index]))
                        self.assertTrue(np.isnan(ds_stds[index]))
                        # original stats are also nan
                        np.testing.assert_almost_equal(
                            standard_ds.mean[index], df[feature].mean()
                        )
                        np.testing.assert_almost_equal(
                            standard_ds.std[index], df[feature].std(ddof=0)
                        )
                        continue

                    # note some NaN values 'E' still have statistics
                    # TODO our reduce statistics has to cope with some NaN!
                    # until then 'E' fails

                    # check transformed std / mean are 1 and 0 per feature
                    np.testing.assert_almost_equal(ds_means[index], 0)
                    np.testing.assert_almost_equal(ds_stds[index], 1)
                    # external statisic matches internally used statistics
                    np.testing.assert_almost_equal(
                        standard_ds.mean[index], df[feature].mean()
                    )
                    np.testing.assert_almost_equal(
                        standard_ds.std[index], df[feature].std(ddof=0)
                    )
                    # order of reduced means matches order in

    def test_processing_parameters_minmax(self) -> None:
        with TestFileContent(self.content) as a_test_file, TestFileContent(
            self.other_content
        ) as another_test_file:
            minmax_dataset = GeneExpressionDataset(
                a_test_file.filename,
                another_test_file.filename,
                backend=self.backend,
                index_col=0,
                standardize=False,
                min_max=True,
            )
            gene_list = minmax_dataset.gene_list
            df = pd.concat(
                [
                    pd.read_csv(a_test_file.filename, index_col=0),
                    pd.read_csv(another_test_file.filename, index_col=0),
                ],
                sort=False,
            )[gene_list]

            # with min max scaling we can check for values 0 and 1
            maximum_array = df.max().values
            minimum_array = df.min().values

            # scalar scaling (single max and min)
            max_n, max_p = map(int, np.unravel_index(np.argmax(df.values), df.shape))
            min_n, min_p = map(int, np.unravel_index(np.argmin(df.values), df.shape))
            for maximum, minimum in [
                # scalar
                [np.max(maximum_array), np.min(minimum_array)],
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
                [maximum_array.tolist(), minimum_array.tolist()],
                # ndarray
                [maximum_array, minimum_array],
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
                    index_col=0,
                )
                data_loader = DataLoader(
                    gene_expression_dataset, batch_size=2, shuffle=True
                )
                for batch_index, batch in enumerate(data_loader):
                    self.assertEqual(
                        batch.shape,
                        (
                            1 if batch_index == 3 else 2,
                            gene_expression_dataset.number_of_features,
                        ),
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
                    index_col=0,
                )
                gene_expression_ds_0 = GeneExpressionDataset(
                    a_test_file.filename, backend=self.backend, index_col=0
                )
                gene_expression_ds_1 = GeneExpressionDataset(
                    another_test_file.filename, backend=self.backend, index_col=0
                )
        all_keys = [
            row.split(',')[0]
            for row in self.content.split(os.linesep)[1:]
            + self.other_content.split(os.linesep)[1:]
        ]

        for ds, keys in [
            (gene_expression_dataset, all_keys),
            (gene_expression_ds_0, all_keys[: len(gene_expression_ds_0)]),
            (gene_expression_ds_1, all_keys[len(gene_expression_ds_0) :]),
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
                index_col=0,
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
