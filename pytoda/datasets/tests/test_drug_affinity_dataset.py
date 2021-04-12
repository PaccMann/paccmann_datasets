"""Testing DrugAffinityDataset."""
import os
import unittest

import numpy as np
from torch.utils.data import DataLoader

from pytoda.datasets import DrugAffinityDataset
from pytoda.tests.utils import TestFileContent

COLUMN_NAMES = [',ligand_name,sequence_id,label', ',drug,protein,class']
DRUG_AFFINITY_CONTENT = os.linesep.join(
    [
        '0,CHEMBL14688,name=20S proteasome chymotrypsin-like-organism=Homo sapiens,1',  # noqa
        '1,CHEMBL14688,name=21S proteasome chymotrypsin-like-organism=Homo sapiens,0',  # noqa
        '2,CHEMBL17564,name=20S proteasome chymotrypsin-like-organism=Homo sapiens,0',  # noqa
        '3,CHEMBL17564,name=21S proteasome chymotrypsin-like-organism=Homo sapiens,1',  # noqa
    ]
)
SMILES_CONTENT = os.linesep.join(
    ['CCO	CHEMBL545', 'C	CHEMBL17564', 'CO	CHEMBL14688', 'NCCS	CHEMBL602']
)
PROTEIN_SEQUENCE_CONTENT = os.linesep.join(
    [
        'ABC\tname=20S proteasome chymotrypsin-like-organism=Homo sapiens',  # noqa
        'DEFG\tname=21S proteasome chymotrypsin-like-organism=Homo sapiens',  # noqa
        'CDEF\tname=22S proteasome chymotrypsin-like-organism=Homo sapiens',  # noqa
        'XZSDASDFF\tname=23S proteasome chymotrypsin-like-organism=Homo sapiens',  # noqa
    ]
)


class TestDrugAffinityDatasetEagerBackend(unittest.TestCase):
    """Testing DrugAffinityDataset with eager backend."""

    def setUp(self):
        self.backend = 'eager'
        print(f'backend is {self.backend}')
        self.smiles_content = SMILES_CONTENT
        self.protein_sequence_content = PROTEIN_SEQUENCE_CONTENT

        for column_names in COLUMN_NAMES:
            self.drug_affinity_content = os.linesep.join(
                [column_names, DRUG_AFFINITY_CONTENT]
            )

            with TestFileContent(self.drug_affinity_content) as drug_affinity_file:
                with TestFileContent(self.smiles_content) as smiles_file:
                    with TestFileContent(
                        self.protein_sequence_content
                    ) as protein_sequence_file:
                        self.drug_affinity_dataset = DrugAffinityDataset(
                            drug_affinity_file.filename,
                            smiles_file.filename,
                            protein_sequence_file.filename,
                            backend=self.backend,
                            column_names=column_names.split(',')[1:],
                        )

    def test___len__(self) -> None:
        """Test __len__."""
        self.assertEqual(len(self.drug_affinity_dataset), 4)

    def test___getitem__(self) -> None:
        """Test __getitem__."""
        smiles_padding_index = (
            self.drug_affinity_dataset.smiles_dataset.smiles_language.padding_index
        )
        smiles_c_index = (
            self.drug_affinity_dataset.smiles_dataset.smiles_language.token_to_index[
                'C'
            ]
        )
        smiles_o_index = (
            self.drug_affinity_dataset.smiles_dataset.smiles_language.token_to_index[
                'O'
            ]
        )
        protein_sequence_padding_index = (
            self.drug_affinity_dataset.protein_sequence_dataset.protein_language.padding_index
        )
        protein_sequence_a_index = self.drug_affinity_dataset.protein_sequence_dataset.protein_language.token_to_index[
            'A'
        ]
        protein_sequence_b_index = self.drug_affinity_dataset.protein_sequence_dataset.protein_language.token_to_index[
            'B'
        ]
        protein_sequence_c_index = self.drug_affinity_dataset.protein_sequence_dataset.protein_language.token_to_index[
            'C'
        ]
        (
            smiles_indexes_tensor,
            protein_sequence_indexes_tensor,
            label_tensor,
        ) = self.drug_affinity_dataset[0]
        np.testing.assert_almost_equal(
            smiles_indexes_tensor.numpy(),
            np.array(
                [
                    smiles_padding_index,
                    smiles_padding_index,
                    smiles_c_index,
                    smiles_o_index,
                ]
            ),
        )
        np.testing.assert_almost_equal(
            protein_sequence_indexes_tensor.numpy(),
            np.array(
                6 * [protein_sequence_padding_index]
                + [
                    protein_sequence_a_index,
                    protein_sequence_b_index,
                    protein_sequence_c_index,
                ]
            ),
        )
        np.testing.assert_almost_equal(label_tensor.numpy(), np.array([1], dtype=int))

    def test_data_loader(self) -> None:
        """Test data_loader."""
        data_loader = DataLoader(self.drug_affinity_dataset, batch_size=2, shuffle=True)
        for (
            batch_index,
            (smiles_indexes_batch, protein_sequence_indexes_batch, label_batch),
        ) in enumerate(data_loader):
            self.assertEqual(smiles_indexes_batch.size(), (2, 4))
            self.assertEqual(protein_sequence_indexes_batch.size(), (2, 9))
            self.assertEqual(label_batch.size(), (2, 1))
            if batch_index > 4:
                break


class TestDrugAffinityDatasetLazyBackend(TestDrugAffinityDatasetEagerBackend):
    """Testing DrugAffinityDataset with lazy backend."""

    def setUp(self):
        self.backend = 'lazy'
        print(f'backend is {self.backend}')
        self.smiles_content = SMILES_CONTENT
        self.protein_sequence_content = PROTEIN_SEQUENCE_CONTENT

        for column_names in COLUMN_NAMES:
            self.drug_affinity_content = os.linesep.join(
                [column_names, DRUG_AFFINITY_CONTENT]
            )

            with TestFileContent(self.drug_affinity_content) as drug_affinity_file:
                with TestFileContent(self.smiles_content) as smiles_file:
                    with TestFileContent(
                        self.protein_sequence_content
                    ) as protein_sequence_file:
                        self.drug_affinity_dataset = DrugAffinityDataset(
                            drug_affinity_file.filename,
                            smiles_file.filename,
                            protein_sequence_file.filename,
                            backend=self.backend,
                            column_names=column_names.split(',')[1:],
                        )


if __name__ == '__main__':
    unittest.main()
