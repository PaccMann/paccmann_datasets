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
        return self.datasource.iloc[index]  # .values

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
    """NOT implementing methods (also not from inheritance)"""
    def __init__(self, data):
        super().__init__()
        self.dataset = data


class TestBaseDatasets(unittest.TestCase):
    """Testing SMILES dataset with eager backend."""

    def random_data(self, length, dims):
        an_array = np.random.randn(self.length, self.dims)
        keys = [uuid.uuid4() for _ in range(self.length)]
        a_df = pd.DataFrame(an_array, index=keys)
        return an_array, A(an_array), keys, a_df, IndexedA(a_df)

    def setUp(self):
        self.length = 10
        self.dims = 5
        (
            self.an_array, self.a_dataset,
            self.keys, self.a_df, self.a_idx_dataset
        ) = self.random_data(self.length, self.dims)
        (
            _, _,
            _, self.a_2nd_df, self.a_2nd_idx_dataset
        ) = self.random_data(self.length, self.dims)
        (
            _, _,
            _, self.a_3rd_df, self.a_3rd_idx_dataset
        ) = self.random_data(self.length, self.dims)

        self.delegating_dataset = Delegating(self.a_2nd_idx_dataset)

        # IndexedDataset.__add__
        self.concat_dataset = self.delegating_dataset + self.a_3rd_idx_dataset

    def test___len__(self) -> None:
        """Test __len__."""
        self.assertEqual(len(self.a_dataset), self.length)
        self.assertEqual(len(self.a_idx_dataset), self.length)
        self.assertEqual(len(self.a_2nd_idx_dataset), self.length)
        self.assertEqual(len(self.a_3rd_idx_dataset), self.length)
        self.assertEqual(len(self.delegating_dataset), self.length)
        self.assertEqual(len(self.concat_dataset), 2*self.length)

    def test___getitem__(self) -> None:
        """Test __getitem__."""
        # Tracing(self.a_3rd_df)[0]

        i = 0
        self.assertTrue(all(self.a_dataset[i] == self.an_array[i]))
        self.assertTrue(all(self.a_idx_dataset[i] == self.a_df.iloc[i]))
        self.assertTrue(all(self.a_2nd_idx_dataset[i] == self.a_2nd_df.iloc[i]))
        self.assertTrue(all(self.a_3rd_idx_dataset[i] == self.a_3rd_df.iloc[i]))
        self.assertTrue(all(self.delegating_dataset[i] == self.a_2nd_df.iloc[i]))
        # first in datasets
        self.assertTrue(all(self.concat_dataset[i] == self.a_2nd_df.iloc[i]))

        i = -1
        self.assertTrue(all(self.a_dataset[i] == self.an_array[i]))
        self.assertTrue(all(self.a_idx_dataset[i] == self.a_df.iloc[i]))
        self.assertTrue(all(self.a_2nd_idx_dataset[i] == self.a_2nd_df.iloc[i]))
        self.assertTrue(all(self.a_3rd_idx_dataset[i] == self.a_3rd_df.iloc[i]))
        self.assertTrue(all(self.delegating_dataset[i] == self.a_2nd_df.iloc[i]))
        # last in datasets
        self.assertTrue(all(self.concat_dataset[i] == self.a_3rd_df.iloc[i]))

    def test_data_loader(self) -> None:
        """Test data_loader."""
        # content = os.linesep.join(
        #     [
        #         'CCO	CHEMBL545',
        #         'C	CHEMBL17564',
        #         'CO	CHEMBL14688',
        #         'NCCS	CHEMBL602',
        #     ]
        # )
        # with TestFileContent(content) as a_test_file:
        #     with TestFileContent(content) as another_test_file:
        #         smiles_dataset = SMILESDataset(
        #             a_test_file.filename,
        #             another_test_file.filename,
        #             backend='eager'
        #         )
        #         data_loader = DataLoader(
        #             smiles_dataset, batch_size=4, shuffle=True
        #         )
        #         for batch_index, batch in enumerate(data_loader):
        #             self.assertEqual(batch.shape, (4, 4))
        #             if batch_index > 10:
        #                 break
        pass


if __name__ == '__main__':
    unittest.main()
