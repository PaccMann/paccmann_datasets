"""Testing basic ways to setup a dataset."""
import unittest
import traceback
import uuid
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytoda.datasets import (
    IndexedDataset, DatasetDelegator,
    _ConcatenatedDataset
)
from pytoda.tests.utils import TestFileContent
from pytoda.types import List, Tuple, Hashable, FileList


class IndexedA(IndexedDataset):
    def __init__(self, df):
        self.datasource = df
        self.range_index = pd.RangeIndex(0, self.__len__())
        super().__init__()

    def __len__(self):
        return len(self.datasource)

    def __getitem__(self, index):
        return self.datasource.iloc[index].values

    def get_key(self, index: int) -> Hashable:
        """Get sample identifier from integer index."""
        return self.datasource.index[index]

    def get_index(self, key: Hashable) -> int:
        """Get index for first datum mapping to the given sample identifier."""
        # item will raise if not single value (deprecated in pandas)
        try:
            index = self.range_index[
                self.datasource.index == key
            ]
            return index.values.item()
        except ValueError:
            if len(index) == 0:
                raise KeyError
            else:
                # key not unique, return first as _ConcatenatedDataset
                return index.values[0]


class Delegating(DatasetDelegator):
    """NOT implementing methods (and only built-ins from inheritance)"""
    def __init__(self, data):
        self.dataset = data


class TestBaseDatasets(unittest.TestCase):
    """Testing SMILES dataset with eager backend."""
    length = 11
    dims = 5

    def random_data(self, length, dims):
        an_array = np.random.randn(self.length, self.dims)
        keys = [str(uuid.uuid4()) for _ in range(self.length)]
        a_df = pd.DataFrame(an_array, index=keys)
        return keys, a_df, IndexedA(a_df)

    def setUp(self):
        (
            self.a_1st_keys, self.a_1st_df, self.a_1st_ds
        ) = self.random_data(self.length, self.dims)
        (
            self.a_2nd_keys, self.a_2nd_df, self.a_2nd_ds
        ) = self.random_data(self.length, self.dims)

        self.delegating_ds = Delegating(self.a_1st_ds)

        # IndexedDataset.__add__ i.e. _ConcatenatedDataset
        self.concat_ds = self.delegating_ds + self.a_2nd_ds
        self.concat_keys = self.a_1st_keys + self.a_2nd_keys

    def test_delegation_dir(self):
        ds_dir = dir(self.delegating_ds)
        # delegated to IndexedA
        self.assertIn('get_key', ds_dir)
        self.assertIn('get_index', ds_dir)
        # delegated to IndexedDataset
        self.assertIn('get_item_from_key', ds_dir)
        self.assertIn('keys', ds_dir)
        # futile, as built-ins delegation needed hardcoding in DatasetDelegator
        self.assertIn('__len__', ds_dir)  # see test___len__
        self.assertIn('__getitem__', ds_dir)  # see test___getitem__
        self.assertIn('__add__', ds_dir)  # see tests on self.concat_ds

        # implementation specific
        # self.assertIn('datasource', ds_dir)
        # self.assertIn('range_index', ds_dir)

    def test___len__(self) -> None:
        """Test __len__."""
        self.assertEqual(len(self.a_1st_ds), self.length)
        self.assertEqual(len(self.a_2nd_ds), self.length)
        self.assertEqual(len(self.delegating_ds), self.length)
        self.assertEqual(len(self.concat_ds), 2*self.length)

    def test___getitem__(self) -> None:
        """Test __getitem__."""
        # Tracing(self.a_2nd_df)[0]

        i = 0
        self.assertTrue(all(self.a_1st_ds[i] == self.a_1st_df.iloc[i]))
        self.assertTrue(all(self.a_2nd_ds[i] == self.a_2nd_df.iloc[i]))
        self.assertTrue(all(self.delegating_ds[i] == self.a_1st_df.iloc[i]))
        # first in datasets
        self.assertTrue(all(self.concat_ds[i] == self.a_1st_df.iloc[i]))

        i = -1
        self.assertTrue(all(self.a_1st_ds[i] == self.a_1st_df.iloc[i]))
        self.assertTrue(all(self.a_2nd_ds[i] == self.a_2nd_df.iloc[i]))
        self.assertTrue(all(self.delegating_ds[i] == self.a_1st_df.iloc[i]))
        # last in datasets
        self.assertTrue(all(self.concat_ds[i] == self.a_2nd_df.iloc[i]))

    def test_data_loader(self) -> None:
        """Test data_loader."""

        batch_size = 4
        a_1st_dl = DataLoader(
            self.a_1st_ds, batch_size=batch_size, shuffle=True, drop_last=False
        )
        full_batches = self.length // batch_size

        for batch_index, batch in enumerate(a_1st_dl):
            if batch_index >= full_batches:  # if drop_last
                self.assertEqual(
                    batch.shape, (self.length % batch_size, self.dims)
                )
            else:
                self.assertEqual(batch.shape, (batch_size, self.dims))

        # concatenated
        concat_dl = DataLoader(
            self.concat_ds, batch_size=batch_size, shuffle=True, drop_last=False
        )
        full_batches = (2 * self.length) // batch_size

        for batch_index, batch in enumerate(concat_dl):
            if batch_index >= full_batches:  # if drop_last
                self.assertEqual(
                    batch.shape, ((2 * self.length) % batch_size, self.dims)
                )
            else:
                self.assertEqual(batch.shape, (batch_size, self.dims))

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

    def test_all_base_for_indexed_methods(self):
        (
            other_keys, _, other_ds
        ) = self.random_data(self.length, self.dims)

        for ds, keys in [
            (self.a_1st_ds, self.a_1st_keys),
            (self.a_2nd_ds, self.a_2nd_keys),
            (self.delegating_ds, self.a_1st_keys),
            (self.concat_ds, self.concat_keys),
        ]:
            index = -1
            self._test_indexed(ds, keys, index)

            # again with self delegation and concatenation
            ds = _ConcatenatedDataset([Delegating(other_ds), Delegating(ds)])
            index = self.length + 1  # dataset_index == 1
            keys = other_keys + keys
            self._test_indexed(ds, keys, index)
            # get_index_pair
            self.assertTupleEqual(
                (1, index-self.length),
                ds.get_index_pair(index)
            )
            # get_key_pair
            self.assertTupleEqual(
                (1, keys[index]),
                ds.get_key_pair(index)
            )
            # _ConcatenatedDataset is not a DatasetDelegator
            self.assertNotIn('datasource', dir(ds))
        
        # duplicate keys lookup returns first
        index == self.length + 1
        duplicate_ds = other_ds + other_ds
        self.assertNotEqual(
            index,
            duplicate_ds.get_index(duplicate_ds.get_key(index))
        )


if __name__ == '__main__':
    unittest.main()
