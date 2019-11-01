"""Testing SMILES dataset with lazy backend."""
import unittest
import os
import numpy as np
from torch.utils.data import DataLoader
from pytoda.datasets import SMILESDataset
from pytoda.tests.utils import TestFileContent


class TestSMILESDatasetLazyBackend(unittest.TestCase):
    """Testing SMILES dataset with lazy backend."""

    def test___len__(self) -> None:
        """Test __len__."""
        content = os.linesep.join(
            [
                'CCO	CHEMBL545',
                'C	CHEMBL17564',
                'CO	CHEMBL14688',
                'NCCS	CHEMBL602',
            ]
        )
        with TestFileContent(content) as a_test_file:
            with TestFileContent(content) as another_test_file:
                smiles_dataset = SMILESDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    backend='lazy'
                )
                self.assertEqual(len(smiles_dataset), 8)

    def test___getitem__(self) -> None:
        """Test __getitem__."""
        content = os.linesep.join(
            [
                'CCO	CHEMBL545',
                'C	CHEMBL17564',
                'CO	CHEMBL14688',
                'NCCS	CHEMBL602',
            ]
        )
        with TestFileContent(content) as a_test_file:
            with TestFileContent(content) as another_test_file:
                smiles_dataset = SMILESDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    backend='lazy'
                )
                padding_index = smiles_dataset.smiles_language.padding_index
                start_index = smiles_dataset.smiles_language.start_index
                stop_index = smiles_dataset.smiles_language.stop_index
                c_index = smiles_dataset.smiles_language.token_to_index['C']
                o_index = smiles_dataset.smiles_language.token_to_index['O']
                n_index = smiles_dataset.smiles_language.token_to_index['N']
                s_index = smiles_dataset.smiles_language.token_to_index['S']
                self.assertListEqual(
                    smiles_dataset[0].numpy().flatten().tolist(),
                    [padding_index, c_index, c_index, o_index]
                )
                self.assertListEqual(
                    smiles_dataset[7].numpy().flatten().tolist(),
                    [n_index, c_index, c_index, s_index]
                )
                smiles_dataset = SMILESDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    padding=False,
                    backend='lazy'
                )
                self.assertListEqual(
                    smiles_dataset[0].numpy().flatten().tolist(),
                    [c_index, c_index, o_index]
                )
                self.assertListEqual(
                    smiles_dataset[7].numpy().flatten().tolist(),
                    [n_index, c_index, c_index, s_index]
                )
                smiles_dataset = SMILESDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    add_start_and_stop=True,
                    backend='lazy'
                )
                self.assertListEqual(
                    smiles_dataset[0].numpy().flatten().tolist(), [
                        padding_index, start_index, c_index, c_index, o_index,
                        stop_index
                    ]
                )
                self.assertListEqual(
                    smiles_dataset[7].numpy().flatten().tolist(), [
                        start_index, n_index, c_index, c_index, s_index,
                        stop_index
                    ]
                )
                smiles_dataset = SMILESDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    augment=True,
                    backend='lazy'
                )
                np.random.seed(0)
                for randomized_smiles in [
                    'CSCN', 'NCCS', 'SCCN', 'CNCS', 'CCSN'
                ]:
                    token_indexes = (
                        smiles_dataset[3].numpy().flatten().tolist()
                    )
                    smiles = (
                        smiles_dataset.smiles_language.
                        token_indexes_to_smiles(token_indexes)
                    )
                    self.assertEqual(smiles, randomized_smiles)

    def test_data_loader(self) -> None:
        """Test data_loader."""
        content = os.linesep.join(
            [
                'CCO	CHEMBL545',
                'C	CHEMBL17564',
                'CO	CHEMBL14688',
                'NCCS	CHEMBL602',
            ]
        )
        with TestFileContent(content) as a_test_file:
            with TestFileContent(content) as another_test_file:
                smiles_dataset = SMILESDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    backend='lazy'
                )
                data_loader = DataLoader(
                    smiles_dataset, batch_size=4, shuffle=True
                )
                for batch_index, batch in enumerate(data_loader):
                    self.assertEqual(batch.shape, (4, 4, 1))
                    if batch_index > 10:
                        break


if __name__ == '__main__':
    unittest.main()
