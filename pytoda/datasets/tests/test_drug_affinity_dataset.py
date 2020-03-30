"""Testing DrugAffinityDataset."""
import unittest
import os
import numpy as np
from torch.utils.data import DataLoader
from pytoda.datasets import DrugAffinityDataset
from pytoda.tests.utils import TestFileContent


class TestDrugAffinityDataset(unittest.TestCase):
    """Testing DrugAffinityDataset with eager backend."""

    def test___len__(self) -> None:
        """Test __len__."""
        drug_affinity_content = os.linesep.join(
            [
                ',ligand_name,sequence_id,label',
                '0,CHEMBL14688,name=20S proteasome chymotrypsin-like-organism=Homo sapiens,1',  # noqa
                '1,CHEMBL14688,name=21S proteasome chymotrypsin-like-organism=Homo sapiens,0',  # noqa
                '2,CHEMBL17564,name=20S proteasome chymotrypsin-like-organism=Homo sapiens,0',  # noqa
                '3,CHEMBL17564,name=21S proteasome chymotrypsin-like-organism=Homo sapiens,1',  # noqa
            ]
        )
        smiles_content = os.linesep.join(
            [
                'CCO	CHEMBL545',
                'C	CHEMBL17564',
                'CO	CHEMBL14688',
                'NCCS	CHEMBL602',
            ]
        )
        protein_sequence_content = os.linesep.join(
            [
                'ABC\tname=20S proteasome chymotrypsin-like-organism=Homo sapiens',  # noqa
                'DEFG\tname=21S proteasome chymotrypsin-like-organism=Homo sapiens',  # noqa
                'CDEF\tname=22S proteasome chymotrypsin-like-organism=Homo sapiens',  # noqa
                'XZSDASDFF\tname=23S proteasome chymotrypsin-like-organism=Homo sapiens',  # noqa
            ]
        )
        with TestFileContent(drug_affinity_content) as drug_affinity_file:
            with TestFileContent(smiles_content) as smiles_file:
                with TestFileContent(
                    protein_sequence_content
                ) as protein_sequence_file:
                    drug_affinity_dataset = DrugAffinityDataset(
                        drug_affinity_file.filename,
                        smiles_file.filename,
                        protein_sequence_file.filename,
                        backend='eager'
                    )
                    self.assertEqual(len(drug_affinity_dataset), 4)

    def test___getitem__(self) -> None:
        """Test __getitem__."""
        drug_affinity_content = os.linesep.join(
            [
                ',ligand_name,sequence_id,label',
                '0,CHEMBL14688,name=20S proteasome chymotrypsin-like-organism=Homo sapiens,1',  # noqa
                '1,CHEMBL14688,name=21S proteasome chymotrypsin-like-organism=Homo sapiens,0',  # noqa
                '2,CHEMBL17564,name=20S proteasome chymotrypsin-like-organism=Homo sapiens,0',  # noqa
                '3,CHEMBL17564,name=21S proteasome chymotrypsin-like-organism=Homo sapiens,1',  # noqa
            ]
        )
        smiles_content = os.linesep.join(
            [
                'CCO	CHEMBL545',
                'C	CHEMBL17564',
                'CO	CHEMBL14688',
                'NCCS	CHEMBL602',
            ]
        )
        protein_sequence_content = os.linesep.join(
            [
                'ABC\tname=20S proteasome chymotrypsin-like-organism=Homo sapiens',  # noqa
                'DEFG\tname=21S proteasome chymotrypsin-like-organism=Homo sapiens',  # noqa
                'CDEF\tname=22S proteasome chymotrypsin-like-organism=Homo sapiens',  # noqa
                'XZSDASDFF\tname=23S proteasome chymotrypsin-like-organism=Homo sapiens',  # noqa
            ]
        )
        with TestFileContent(drug_affinity_content) as drug_affinity_file:
            with TestFileContent(smiles_content) as smiles_file:
                with TestFileContent(
                    protein_sequence_content
                ) as protein_sequence_file:
                    drug_affinity_dataset = DrugAffinityDataset(
                        drug_affinity_file.filename,
                        smiles_file.filename,
                        protein_sequence_file.filename,
                        backend='eager'
                    )
                    smiles_padding_index = (
                        drug_affinity_dataset.smiles_dataset.smiles_language.
                        padding_index
                    )
                    smiles_c_index = (
                        drug_affinity_dataset.smiles_dataset.smiles_language.
                        token_to_index['C']
                    )
                    smiles_o_index = (
                        drug_affinity_dataset.smiles_dataset.smiles_language.
                        token_to_index['O']
                    )
                    protein_sequence_padding_index = (
                        drug_affinity_dataset.protein_sequence_dataset.
                        protein_language.padding_index
                    )
                    protein_sequence_a_index = (
                        drug_affinity_dataset.protein_sequence_dataset.
                        protein_language.token_to_index['A']
                    )
                    protein_sequence_b_index = (
                        drug_affinity_dataset.protein_sequence_dataset.
                        protein_language.token_to_index['B']
                    )
                    protein_sequence_c_index = (
                        drug_affinity_dataset.protein_sequence_dataset.
                        protein_language.token_to_index['C']
                    )
                    (
                        smiles_indexes_tensor, protein_sequence_indexes_tensor,
                        label_tensor
                    ) = drug_affinity_dataset[0]
                    np.testing.assert_almost_equal(
                        smiles_indexes_tensor.numpy(),
                        np.array(
                            [
                                smiles_padding_index, smiles_padding_index,
                                smiles_c_index, smiles_o_index
                            ]
                        )
                    )
                    np.testing.assert_almost_equal(
                        protein_sequence_indexes_tensor.numpy(),
                        np.array(
                            6*[protein_sequence_padding_index] +
                            [
                                protein_sequence_a_index,
                                protein_sequence_b_index,
                                protein_sequence_c_index
                            ]
                        )
                    )
                    np.testing.assert_almost_equal(
                        label_tensor.numpy(), np.array([1], dtype=int)
                    )

    def test_data_loader(self) -> None:
        """Test data_loader."""
        drug_affinity_content = os.linesep.join(
            [
                ',ligand_name,sequence_id,label',
                '0,CHEMBL14688,name=20S proteasome chymotrypsin-like-organism=Homo sapiens,1',  # noqa
                '1,CHEMBL14688,name=21S proteasome chymotrypsin-like-organism=Homo sapiens,0',  # noqa
                '2,CHEMBL17564,name=20S proteasome chymotrypsin-like-organism=Homo sapiens,0',  # noqa
                '3,CHEMBL17564,name=21S proteasome chymotrypsin-like-organism=Homo sapiens,1',  # noqa
            ]
        )
        smiles_content = os.linesep.join(
            [
                'CCO	CHEMBL545',
                'C	CHEMBL17564',
                'CO	CHEMBL14688',
                'NCCS	CHEMBL602',
            ]
        )
        protein_sequence_content = os.linesep.join(
            [
                'ABC\tname=20S proteasome chymotrypsin-like-organism=Homo sapiens',  # noqa
                'DEFG\tname=21S proteasome chymotrypsin-like-organism=Homo sapiens',  # noqa
                'CDEF\tname=22S proteasome chymotrypsin-like-organism=Homo sapiens',  # noqa
                'XZSDASDFF\tname=23S proteasome chymotrypsin-like-organism=Homo sapiens',  # noqa
            ]
        )
        with TestFileContent(drug_affinity_content) as drug_affinity_file:
            with TestFileContent(smiles_content) as smiles_file:
                with TestFileContent(
                    protein_sequence_content
                ) as protein_sequence_file:
                    drug_affinity_dataset = DrugAffinityDataset(
                        drug_affinity_file.filename,
                        smiles_file.filename,
                        protein_sequence_file.filename,
                        backend='eager'
                    )
                    data_loader = DataLoader(
                        drug_affinity_dataset, batch_size=2, shuffle=True
                    )
                    for (
                        batch_index, (
                            smiles_indexes_batch,
                            protein_sequence_indexes_batch, label_batch
                        )
                    ) in enumerate(data_loader):
                        self.assertEqual(smiles_indexes_batch.size(), (2, 4))
                        self.assertEqual(
                            protein_sequence_indexes_batch.size(), (2, 9)
                        )
                        self.assertEqual(label_batch.size(), (2, 1))
                        if batch_index > 4:
                            break


if __name__ == '__main__':
    unittest.main()
