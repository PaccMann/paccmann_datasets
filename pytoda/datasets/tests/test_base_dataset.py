"""Testing basic ways to setup a dataset."""
import unittest
import uuid
from copy import copy

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from pytoda.datasets import (
    ConcatKeyDataset,
    DatasetDelegator,
    KeyDataset,
    indexed,
    keyed,
)
from pytoda.types import Hashable


class Indexed(KeyDataset):
    """As DataFrameDataset but only implementing necessary methods."""

    def __init__(self, df):
        self.df = df
        super().__init__()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.df.iloc[index].values

    def get_key(self, index: int) -> Hashable:
        """Get key from integer index."""
        return self.df.index[index]

    def get_index(self, key: Hashable) -> int:
        """Get index for first datum mapping to the given key."""
        # item will raise if not single value (deprecated in pandas)
        try:
            indexes = np.nonzero(self.df.index == key)[0]
            return indexes.item()
        except ValueError:
            if len(indexes) == 0:
                raise KeyError
            else:
                # key not unique, return first as ConcatKeyDataset
                return indexes[0]


class Delegating(DatasetDelegator):
    """NOT implementing methods (and only built-ins from inheritance)."""

    def __init__(self, data):
        self.dataset = data


class TestBaseDatasets(unittest.TestCase):
    """Testing dataset for base methods."""

    length = 11  # of a single dataset
    dims = 5

    def random_data(self, length, dims):
        an_array = np.random.randn(self.length, self.dims)
        keys = [str(uuid.uuid4()) for _ in range(self.length)]
        a_df = pd.DataFrame(an_array, index=keys)
        return keys, a_df, Indexed(a_df)

    def setUp(self):
        (self.a_1st_keys, self.a_1st_df, self.a_1st_ds) = self.random_data(
            self.length, self.dims
        )
        (self.a_2nd_keys, self.a_2nd_df, self.a_2nd_ds) = self.random_data(
            self.length, self.dims
        )

        self.delegating_ds = Delegating(self.a_1st_ds)

        # KeyDataset.__add__ i.e. ConcatKeyDataset
        self.concat_ds = self.delegating_ds + self.a_2nd_ds
        self.concat_keys = self.a_1st_keys + self.a_2nd_keys

    def assertListedEqual(self, listable1, listable2):
        """Easier comparison between arrays, series and or lists."""
        self.assertListEqual(list(listable1), list(listable2))

    def test_delegation_dir(self):
        # stacking delegation
        ds_dir = dir(Delegating(Delegating(self.delegating_ds)))
        # delegated to Indexed
        self.assertIn('get_key', ds_dir)
        self.assertIn('get_index', ds_dir)
        # delegated to KeyDataset
        self.assertIn('get_item_from_key', ds_dir)
        self.assertIn('keys', ds_dir)
        self.assertIn('has_duplicate_keys', ds_dir)
        # futile, as built-ins delegation needed hardcoding in DatasetDelegator
        self.assertIn('__len__', ds_dir)  # see test___len__
        self.assertIn('__getitem__', ds_dir)  # see test___getitem__
        self.assertIn('__add__', ds_dir)  # see tests on self.concat_ds

        # no tests on implementation specific attributes here

    def test___len__(self) -> None:
        """Test __len__."""
        self.assertEqual(len(self.a_1st_ds), self.length)
        self.assertEqual(len(self.a_2nd_ds), self.length)
        self.assertEqual(len(self.delegating_ds), self.length)
        self.assertEqual(len(self.concat_ds), 2 * self.length)

    def test___getitem__(self) -> None:
        """Test __getitem__."""

        i = 0
        self.assertListedEqual(self.a_1st_ds[i], self.a_1st_df.iloc[i])
        self.assertListedEqual(self.a_2nd_ds[i], self.a_2nd_df.iloc[i])
        self.assertListedEqual(self.delegating_ds[i], self.a_1st_df.iloc[i])
        # first in datasets
        self.assertListedEqual(self.concat_ds[i], self.a_1st_df.iloc[i])

        i = -1
        self.assertListedEqual(self.a_1st_ds[i], self.a_1st_df.iloc[i])
        self.assertListedEqual(self.a_2nd_ds[i], self.a_2nd_df.iloc[i])
        self.assertListedEqual(self.delegating_ds[i], self.a_1st_df.iloc[i])
        # last in datasets
        self.assertListedEqual(self.concat_ds[i], self.a_2nd_df.iloc[i])

    def _test__getitem__modified(self, mutate_copy) -> None:
        """Test __getitem__ returning tuple with item first."""
        for i in [0, -1]:
            self.assertListedEqual(
                mutate_copy(self.a_1st_ds)[i][0], self.a_1st_df.iloc[i]
            )
            self.assertListedEqual(
                mutate_copy(self.a_2nd_ds)[i][0], self.a_2nd_df.iloc[i]
            )
            self.assertListedEqual(
                mutate_copy(self.delegating_ds)[i][0], self.a_1st_df.iloc[i]
            )

        # first in datasets
        self.assertListedEqual(mutate_copy(self.concat_ds)[0][0], self.a_1st_df.iloc[0])
        # last in datasets
        self.assertListedEqual(
            mutate_copy(self.concat_ds)[-1][0], self.a_2nd_df.iloc[-1]
        )

    def test__getitem__mutating_utils(self):
        self._test__getitem__modified(mutate_copy=indexed)
        self._test__getitem__modified(mutate_copy=keyed)

    def test_data_loader(self) -> None:
        """Test data_loader."""

        batch_size = 4
        a_1st_dl = DataLoader(self.a_1st_ds, batch_size=batch_size, shuffle=True)
        full_batches = self.length // batch_size

        for batch_index, batch in enumerate(a_1st_dl):
            if batch_index >= full_batches:  # if drop_last
                self.assertEqual(batch.shape, (self.length % batch_size, self.dims))
            else:
                self.assertEqual(batch.shape, (batch_size, self.dims))

        # concatenated
        concat_dl = DataLoader(self.concat_ds, batch_size=batch_size, shuffle=True)
        full_batches = (2 * self.length) // batch_size

        for batch_index, batch in enumerate(concat_dl):
            if batch_index >= full_batches:  # if drop_last
                self.assertEqual(
                    batch.shape, ((2 * self.length) % batch_size, self.dims)
                )
            else:
                self.assertEqual(batch.shape, (batch_size, self.dims))

    def _test_item_independent(self, ds, keys, index):
        key = keys[index]
        positive_index = index % len(ds)
        # get_key (support for negative index?)
        self.assertEqual(key, ds.get_key(positive_index))
        self.assertEqual(key, ds.get_key(index))
        # get_index
        self.assertEqual(positive_index, ds.get_index(key))
        # keys
        self.assertListedEqual(keys, ds.keys())
        # duplicate keys
        self.assertFalse(ds.has_duplicate_keys)

    def _test_base_methods(self, ds, keys, index):
        key = keys[index]
        self._test_item_independent(ds, keys, index)
        # get_item_from_key
        self.assertListedEqual(ds[index], ds.get_item_from_key(key))
        # in case of returning a tuple:
        # for from_index, from_key in zip(ds[index], ds.get_item_from_key(key)):  # noqa
        #     self.assertListedEqual(from_index, from_key)

    def _test_keyed_util(self, ds, keys, index):
        ds_ = keyed(ds)
        key = keys[index]
        self._test_item_independent(ds_, keys, index)

        # modified methods
        item_of_i, k_of_i = ds_[index]
        item_of_k, k_of_k = ds_.get_item_from_key(key)

        # get_item_from_key
        self.assertListedEqual(item_of_i, item_of_k)
        self.assertTrue(key == k_of_i and key == k_of_k)

    def _test_indexed_util(self, ds, keys, index):
        ds_ = indexed(ds)
        key = keys[index]
        positive_index = index % len(ds_)
        self._test_item_independent(ds_, keys, index)

        # modified methods
        item_of_i, i_of_i = ds_[index]
        item_of_k, i_of_k = ds_.get_item_from_key(key)

        # get_item_from_key
        self.assertListedEqual(item_of_i, item_of_k)
        self.assertTrue(positive_index == i_of_i and positive_index == i_of_k)

    def _test_stacked_indexed_keyed_util(self, ds, keys, index):
        ds_ = indexed(keyed(indexed(ds)))
        key = keys[index]
        positive_index = index % len(ds_)
        self._test_item_independent(ds_, keys, index)

        # modified methods
        (((item_of_i, i_of_i0), k_of_i), i_of_i1) = ds_[index]
        (((item_of_k, i_of_k0), k_of_k), i_of_k1) = ds_.get_item_from_key(key)

        # get_item_from_key
        self.assertListedEqual(item_of_i, item_of_k)
        self.assertTrue(key == k_of_i and key == k_of_k)
        self.assertTrue(
            positive_index == i_of_i0
            and positive_index == i_of_i1
            and positive_index == i_of_k0
            and positive_index == i_of_k1
        )

    def test_all_base_for_indexed_methods_and_copy(self):
        (other_keys, _, other_ds) = self.random_data(self.length, self.dims)

        for ds, keys in [
            (self.a_1st_ds, self.a_1st_keys),
            (self.a_2nd_ds, self.a_2nd_keys),
            (self.delegating_ds, self.a_1st_keys),
            (self.concat_ds, self.concat_keys),
            # test shallow copy (not trivial with delegation)
            (copy(self.a_1st_ds), self.a_1st_keys),
            (copy(self.a_2nd_ds), self.a_2nd_keys),
            (copy(self.delegating_ds), self.a_1st_keys),
            (copy(self.concat_ds), self.concat_keys),
        ]:
            index = -1
            self._test_indexed_util(ds, keys, index)
            self._test_keyed_util(ds, keys, index)
            self._test_stacked_indexed_keyed_util(ds, keys, index)
            self._test_base_methods(ds, keys, index)

            # again with self delegation and concatenation
            ds = ConcatKeyDataset([Delegating(other_ds), Delegating(ds)])
            index = self.length + 1  # dataset_index == 1
            keys = other_keys + keys
            self._test_indexed_util(ds, keys, index)
            self._test_keyed_util(ds, keys, index)
            self._test_stacked_indexed_keyed_util(ds, keys, index)
            self._test_base_methods(ds, keys, index)
            # get_index_pair
            self.assertTupleEqual((1, index - self.length), ds.get_index_pair(index))
            # get_key_pair
            self.assertTupleEqual((1, keys[index]), ds.get_key_pair(index))
            # ConcatKeyDataset is not a DatasetDelegator
            self.assertNotIn('df', dir(ds))

        index == self.length + 1
        duplicate_ds = other_ds + other_ds
        self.assertTrue(duplicate_ds.has_duplicate_keys)
        # duplicate keys lookup returns first in this case
        self.assertNotEqual(index, duplicate_ds.get_index(duplicate_ds.get_key(index)))


if __name__ == '__main__':
    unittest.main()
