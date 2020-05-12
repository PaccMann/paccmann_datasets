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


class A(Dataset):
    def __init__(self, array):
        self.datasource = array
        super().__init__()

    def __len__(self):
        return len(self.datasource)

    def __getitem__(self, index):
        return self.datasource[index]


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
        """Get integer index from sample identifier."""
        # item will raise if not single value (deprecated in pandas)
        return self.range_index[self.datasource.index == key].values.item()


class Tracing(IndexedA):
    def __getitem__(self, index):
        traceback.print_stack()
        return self.datasource.iloc[index]


class Delegating(DatasetDelegator):
    """NOT implementing methods (and only built-ins from inheritance)"""
    def __init__(self, data):
        self.dataset = data


class TestBaseDatasets(unittest.TestCase):
    """Testing SMILES dataset with eager backend."""

    def random_data(self, length, dims):
        an_array = np.random.randn(self.length, self.dims)
        keys = [uuid.uuid4() for _ in range(self.length)]
        a_df = pd.DataFrame(an_array, index=keys)
        return an_array, A(an_array), keys, a_df, IndexedA(a_df)

    def setUp(self):
        self.length = 11
        self.dims = 5
        # ds for dataset
        (
            self.an_array, self.a_default_ds,
            self.keys, self.a_1st_df, self.a_1st_ds
        ) = self.random_data(self.length, self.dims)
        (
            _, _,
            _, self.a_2nd_df, self.a_2nd_ds
        ) = self.random_data(self.length, self.dims)
        (
            _, _,
            _, self.a_3rd_df, self.a_3rd_ds
        ) = self.random_data(self.length, self.dims)

        self.delegating_ds = Delegating(self.a_2nd_ds)

        # IndexedDataset.__add__
        self.concat_ds = self.delegating_ds + self.a_3rd_ds

    def test_delegation_dir(self):
        ds_dir = dir(self.delegating_ds)
        # delegated to IndexedA
        self.assertIn('get_key', ds_dir)
        self.assertIn('get_index', ds_dir)
        # delegated to IndexedDataset
        self.assertIn('get_item_from_key', ds_dir)
        self.assertIn('keys', ds_dir)
        # futile, as built-ins delegation needed hardcoding in DatasetDelegator
        # self.assertIn('__len__', ds_dir)  # see test___len__
        # self.assertIn('__getitem__', ds_dir)  # see test___getitem__
        # self.assertIn('__add__', ds_dir)  # see tests on self.concat_ds

    def test___len__(self) -> None:
        """Test __len__."""
        self.assertEqual(len(self.a_default_ds), self.length)
        self.assertEqual(len(self.a_1st_ds), self.length)
        self.assertEqual(len(self.a_2nd_ds), self.length)
        self.assertEqual(len(self.a_3rd_ds), self.length)
        self.assertEqual(len(self.delegating_ds), self.length)
        self.assertEqual(len(self.concat_ds), 2*self.length)

    def test___getitem__(self) -> None:
        """Test __getitem__."""
        # Tracing(self.a_3rd_df)[0]

        i = 0
        self.assertTrue(all(self.a_default_ds[i] == self.an_array[i]))
        self.assertTrue(all(self.a_1st_ds[i] == self.a_1st_df.iloc[i]))
        self.assertTrue(all(self.a_2nd_ds[i] == self.a_2nd_df.iloc[i]))
        self.assertTrue(all(self.a_3rd_ds[i] == self.a_3rd_df.iloc[i]))
        self.assertTrue(all(self.delegating_ds[i] == self.a_2nd_df.iloc[i]))
        # first in datasets
        self.assertTrue(all(self.concat_ds[i] == self.a_2nd_df.iloc[i]))

        i = -1
        self.assertTrue(all(self.a_default_ds[i] == self.an_array[i]))
        self.assertTrue(all(self.a_1st_ds[i] == self.a_1st_df.iloc[i]))
        self.assertTrue(all(self.a_2nd_ds[i] == self.a_2nd_df.iloc[i]))
        self.assertTrue(all(self.a_3rd_ds[i] == self.a_3rd_df.iloc[i]))
        self.assertTrue(all(self.delegating_ds[i] == self.a_2nd_df.iloc[i]))
        # last in datasets
        self.assertTrue(all(self.concat_ds[i] == self.a_3rd_df.iloc[i]))

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


if __name__ == '__main__':
    unittest.main()
