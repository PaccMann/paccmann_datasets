"""Testing SMILES dataset with lazy backend."""
import unittest
import os
import numpy as np
from torch.utils.data import DataLoader
from pytoda.datasets import SMILESDataset
from pytoda.tests.utils import TestFileContent

CONTENT = os.linesep.join(
    [
        'CCO	CHEMBL545',
        'C	CHEMBL17564',
        'CO	CHEMBL14688',
        'NCCS	CHEMBL602',
    ]
)
MORE_CONTENT = os.linesep.join(
    [
        'COCC(C)N	CHEMBL3184692',
        'COCCOC	CHEMBL1232411',
        'O=CC1CCC1	CHEMBL18475',  # longest with length 9
        'NC(=O)O	CHEMBL125278',
    ]
)
LONGEST = 9


class TestSMILESDatasetEagerBackend(unittest.TestCase):
    """Testing SMILES dataset with eager backend."""

    def setUp(self):
        self.backend = 'eager'
        print(f'backend is {self.backend}')
        self.content = CONTENT
        self.other_content = MORE_CONTENT
        self.longest = LONGEST

    def test___len__(self) -> None:
        """Test __len__."""

        with TestFileContent(self.content) as a_test_file:
            with TestFileContent(self.other_content) as another_test_file:
                smiles_dataset = SMILESDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    backend=self.backend
                )
                self.assertEqual(len(smiles_dataset), 8)

    def test___getitem__(self) -> None:
        """Test __getitem__."""

        with TestFileContent(self.content) as a_test_file:
            with TestFileContent(self.other_content) as another_test_file:
                smiles_dataset = SMILESDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    padding=True,
                    augment=False,
                    kekulize=True,
                    sanitize=True,
                    all_bonds_explicit=True,
                    remove_chirality=True,
                    backend=self.backend
                )
                pad_index = smiles_dataset.smiles_language.padding_index
                start_index = smiles_dataset.smiles_language.start_index
                stop_index = smiles_dataset.smiles_language.stop_index
                c_index = smiles_dataset.smiles_language.token_to_index['C']
                o_index = smiles_dataset.smiles_language.token_to_index['O']
                n_index = smiles_dataset.smiles_language.token_to_index['N']
                s_index = smiles_dataset.smiles_language.token_to_index['S']
                d_index = smiles_dataset.smiles_language.token_to_index['-']

                sample = 0
                padding_len = smiles_dataset.padding_length - (
                    len(
                        # str from underlying concatenated _smi dataset
                        smiles_dataset.dataset.dataset[sample]
                    ) * 2 - 1  # just d_index, no '=', '(', etc.
                )
                self.assertListEqual(
                    smiles_dataset[sample].numpy().flatten().tolist(),
                    [pad_index] * padding_len + [
                        c_index, d_index, c_index,
                        d_index, o_index
                    ]
                )

                sample = 3
                padding_len = smiles_dataset.padding_length - (
                    len(
                        # str from underlying concatenated _smi dataset
                        smiles_dataset.dataset.dataset[sample]
                    ) * 2 - 1  # just d_index, no '=', '(', etc.
                )
                self.assertListEqual(
                    smiles_dataset[sample].numpy().flatten().tolist(),
                    [pad_index] * padding_len + [
                        n_index, d_index, c_index, d_index, c_index, d_index,
                        s_index
                    ]
                )

                sample = 5
                padding_len = smiles_dataset.padding_length - (
                    len(
                        # str from underlying concatenated _smi dataset
                        smiles_dataset.dataset.dataset[sample]
                    ) * 2 - 1  # just d_index, no '=', '(', etc.
                )
                self.assertListEqual(
                    smiles_dataset[sample].numpy().flatten().tolist(),
                    [pad_index] * padding_len + [
                        c_index, d_index, o_index, d_index, c_index, d_index,
                        c_index, d_index, o_index, d_index, c_index
                    ]
                )

                smiles_dataset = SMILESDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    padding=False,
                    backend=self.backend
                )
                self.assertListEqual(
                    smiles_dataset[0].numpy().flatten().tolist(),
                    [c_index, c_index, o_index]
                )
                self.assertListEqual(
                    smiles_dataset[5].numpy().flatten().tolist(),
                    [c_index, o_index, c_index, c_index, o_index, c_index]
                )

                smiles_dataset = SMILESDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    add_start_and_stop=True,
                    backend=self.backend
                )
                self.assertEqual(smiles_dataset.padding_length, self.longest+2)

                sample = 0
                padding_len = smiles_dataset.padding_length - (
                    len(
                        # str from underlying concatenated _smi dataset
                        smiles_dataset.dataset.dataset[sample]
                    )
                ) - 2  # start and stop
                self.assertListEqual(
                    smiles_dataset[sample].numpy().flatten().tolist(),
                    [pad_index] * padding_len + [
                        start_index,
                        c_index, c_index, o_index,
                        stop_index
                    ]
                )

                sample = 5
                padding_len = smiles_dataset.padding_length - (
                    len(
                        # str from underlying concatenated _smi dataset
                        smiles_dataset.dataset.dataset[sample]
                    )
                ) - 2  # start and stop
                self.assertListEqual(
                    smiles_dataset[sample].numpy().flatten().tolist(),
                    [pad_index] * padding_len + [
                        start_index,
                        c_index, o_index, c_index, c_index, o_index, c_index,
                        stop_index
                    ]
                )

                smiles_dataset = SMILESDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    augment=True,
                    backend=self.backend
                )
                np.random.seed(0)
                for randomized_smiles in [
                    'C(S)CN', 'NCCS', 'SCCN', 'C(N)CS', 'C(CS)N'
                ]:
                    token_indexes = (
                        smiles_dataset[3].numpy().flatten().tolist()
                    )
                    smiles = (
                        smiles_dataset.smiles_language.
                        token_indexes_to_smiles(token_indexes)
                    )
                    self.assertEqual(smiles, randomized_smiles)

                smiles_dataset = SMILESDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    padding=False,
                    add_start_and_stop=True,
                    remove_bonddir=True,
                    selfies=True,
                    backend=self.backend
                )
                c_index = smiles_dataset.smiles_language.token_to_index['[C]']
                o_index = smiles_dataset.smiles_language.token_to_index['[O]']
                n_index = smiles_dataset.smiles_language.token_to_index['[N]']
                s_index = smiles_dataset.smiles_language.token_to_index['[S]']

                self.assertListEqual(
                    smiles_dataset[0].numpy().flatten().tolist(),
                    [start_index, c_index, c_index, o_index, stop_index]
                )
                self.assertListEqual(
                    smiles_dataset[3].numpy().flatten().tolist(), [
                        start_index, n_index, c_index, c_index, s_index,
                        stop_index
                    ]
                )

    def test_data_loader(self) -> None:
        """Test data_loader."""
        with TestFileContent(self.content) as a_test_file:
            with TestFileContent(self.other_content) as another_test_file:
                smiles_dataset = SMILESDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    backend=self.backend
                )
                data_loader = DataLoader(
                    smiles_dataset, batch_size=4, shuffle=True
                )
                for batch_index, batch in enumerate(data_loader):
                    self.assertEqual(batch.shape, (4, self.longest))
                    if batch_index > 10:
                        break


class TestSMILESDatasetLazyBackend(TestSMILESDatasetEagerBackend):
    """Testing SMILES dataset with lazy backend."""

    def setUp(self):
        self.backend = 'lazy'
        print(f'backend is {self.backend}')
        self.content = CONTENT
        self.other_content = MORE_CONTENT
        self.longest = LONGEST


if __name__ == '__main__':
    unittest.main()
