"""Testing basic ways to setup a dataset."""
import unittest
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
        return self.datasource.iloc[index]

    def get_key(self, index: int) -> Hashable:
        """Get sample identifier from integer index."""
        return self.datasource.index[index]

    def get_index(self, key: Hashable) -> int:
        """Get integer index from sample identifier."""
        return self.range_index[self.datasource.index == key].item()


class Delegating(DatasetDelegator):
    """if NOT implementing abstract methods (also not from inheritance)"""
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
        self.concat_dataset = self.a_3rd_idx_dataset + self.delegating_dataset

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
        #             padding=True,
        #             augment=False,
        #             kekulize=True,
        #             all_bonds_explicit=True,
        #             remove_chirality=True,
        #             backend='eager'
        #         )
        #         pad_index = smiles_dataset.smiles_language.padding_index
        #         start_index = smiles_dataset.smiles_language.start_index
        #         stop_index = smiles_dataset.smiles_language.stop_index
        #         c_index = smiles_dataset.smiles_language.token_to_index['C']
        #         o_index = smiles_dataset.smiles_language.token_to_index['O']
        #         n_index = smiles_dataset.smiles_language.token_to_index['N']
        #         s_index = smiles_dataset.smiles_language.token_to_index['S']
        #         d_index = smiles_dataset.smiles_language.token_to_index['-']

        #         self.assertListEqual(
        #             smiles_dataset[0].numpy().flatten().tolist(), [
        #                 pad_index, pad_index, c_index, d_index, c_index,
        #                 d_index, o_index
        #             ]
        #         )
        #         self.assertListEqual(
        #             smiles_dataset[7].numpy().flatten().tolist(), [
        #                 n_index, d_index, c_index, d_index, c_index, d_index,
        #                 s_index
        #             ]
        #         )
        #         smiles_dataset = SMILESDataset(
        #             a_test_file.filename,
        #             another_test_file.filename,
        #             padding=False,
        #             backend='eager'
        #         )
        #         self.assertListEqual(
        #             smiles_dataset[0].numpy().flatten().tolist(),
        #             [c_index, c_index, o_index]
        #         )
        #         self.assertListEqual(
        #             smiles_dataset[7].numpy().flatten().tolist(),
        #             [n_index, c_index, c_index, s_index]
        #         )
        #         smiles_dataset = SMILESDataset(
        #             a_test_file.filename,
        #             another_test_file.filename,
        #             padding_length=6,
        #             add_start_and_stop=True,
        #             backend='eager'
        #         )
        #         self.assertListEqual(
        #             smiles_dataset[0].numpy().flatten().tolist(), [
        #                 pad_index, start_index, c_index, c_index, o_index,
        #                 stop_index
        #             ]
        #         )
        #         self.assertListEqual(
        #             smiles_dataset[7].numpy().flatten().tolist(), [
        #                 start_index, n_index, c_index, c_index, s_index,
        #                 stop_index
        #             ]
        #         )
        #         smiles_dataset = SMILESDataset(
        #             a_test_file.filename,
        #             another_test_file.filename,
        #             padding_length=8,
        #             augment=True,
        #             backend='eager'
        #         )
        #         np.random.seed(0)
        #         for randomized_smiles in [
        #             'C(S)CN', 'NCCS', 'SCCN', 'C(N)CS', 'C(CS)N'
        #         ]:
        #             token_indexes = (
        #                 smiles_dataset[3].numpy().flatten().tolist()
        #             )

        #             smiles = (
        #                 smiles_dataset.smiles_language.
        #                 token_indexes_to_smiles(token_indexes)
        #             )
        #             self.assertEqual(smiles, randomized_smiles)

        #         smiles_dataset = SMILESDataset(
        #             a_test_file.filename,
        #             another_test_file.filename,
        #             padding=False,
        #             add_start_and_stop=True,
        #             remove_bonddir=True,
        #             selfies=True,
        #             backend='eager'
        #         )
        #         c_index = smiles_dataset.smiles_language.token_to_index['[C]']
        #         o_index = smiles_dataset.smiles_language.token_to_index['[O]']
        #         n_index = smiles_dataset.smiles_language.token_to_index['[N]']
        #         s_index = smiles_dataset.smiles_language.token_to_index['[S]']

        #         self.assertListEqual(
        #             smiles_dataset[0].numpy().flatten().tolist(),
        #             [start_index, c_index, c_index, o_index, stop_index]
        #         )
        #         self.assertListEqual(
        #             smiles_dataset[7].numpy().flatten().tolist(), [
        #                 start_index, n_index, c_index, c_index, s_index,
        #                 stop_index
        #             ]
        #         )
        pass

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
