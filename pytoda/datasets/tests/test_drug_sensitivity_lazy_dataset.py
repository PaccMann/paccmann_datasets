"""Testing DrugSensitivityDataset."""
import unittest
import os
import numpy as np
from torch.utils.data import DataLoader
from pytoda.datasets import DrugSensitivityDataset
from pytoda.tests.utils import TestFileContent


class TestDrugSensitivityDataset(unittest.TestCase):
    """Testing DrugSensitivityDataset with lazy backend."""

    def test___len__(self) -> None:
        """Test __len__."""
        drug_sensitivity_content = os.linesep.join(
            [
                ',drug,cell_line,IC50',
                '0,CHEMBL14688,sample_3,2.1',
                '1,CHEMBL14688,sample_2,-0.9',
                '2,CHEMBL17564,sample_1,1.2',
                '3,CHEMBL17564,sample_2,1.5',
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
        gene_expression_content = os.linesep.join(
            [
                'genes,A,B,C,D',
                'sample_3,9.45,4.984,7.016,8.336',
                'sample_2,7.188,0.695,10.34,6.047',
                'sample_1,9.25,6.133,5.047,5.6',
            ]
        )
        with TestFileContent(
            drug_sensitivity_content
        ) as drug_sensitivity_file:
            with TestFileContent(smiles_content) as smiles_file:
                with TestFileContent(
                    gene_expression_content
                ) as gene_expression_file:
                    drug_sensitivity_dataset = DrugSensitivityDataset(
                        drug_sensitivity_file.filename,
                        smiles_file.filename,
                        gene_expression_file.filename,
                        backend='lazy'
                    )
                    self.assertEqual(len(drug_sensitivity_dataset), 4)

    def test___getitem__(self) -> None:
        """Test __getitem__."""
        drug_sensitivity_content = os.linesep.join(
            [
                ',drug,cell_line,IC50',
                '0,CHEMBL14688,sample_3,2.1',
                '1,CHEMBL14688,sample_2,-0.9',
                '2,CHEMBL17564,sample_1,1.2',
                '3,CHEMBL17564,sample_2,1.5',
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
        gene_expression_content = os.linesep.join(
            [
                'genes,A,B,C,D',
                'sample_3,9.45,4.984,7.016,8.336',
                'sample_2,7.188,0.695,10.34,6.047',
                'sample_1,9.25,6.133,5.047,5.6',
            ]
        )
        with TestFileContent(
            drug_sensitivity_content
        ) as drug_sensitivity_file:
            with TestFileContent(smiles_content) as smiles_file:
                with TestFileContent(
                    gene_expression_content
                ) as gene_expression_file:
                    drug_sensitivity_dataset = DrugSensitivityDataset(
                        drug_sensitivity_file.filename,
                        smiles_file.filename,
                        gene_expression_file.filename,
                        backend='lazy'
                    )
                    padding_index = (
                        drug_sensitivity_dataset.smiles_dataset.
                        smiles_language.padding_index
                    )
                    c_index = (
                        drug_sensitivity_dataset.smiles_dataset.
                        smiles_language.token_to_index['C']
                    )
                    o_index = (
                        drug_sensitivity_dataset.smiles_dataset.
                        smiles_language.token_to_index['O']
                    )
                    (
                        token_indexes_tensor, gene_expression_tensor,
                        ic50_tensor
                    ) = drug_sensitivity_dataset[0]
                    np.testing.assert_almost_equal(
                        token_indexes_tensor.numpy(),
                        np.array(
                            [padding_index, padding_index, c_index, o_index]
                        )
                    )
                    np.testing.assert_almost_equal(
                        gene_expression_tensor.numpy(),
                        drug_sensitivity_dataset.gene_expression_dataset[
                            drug_sensitivity_dataset.gene_expression_dataset.
                            sample_to_index_mapping['sample_3']].numpy()
                    )
                    np.testing.assert_almost_equal(
                        ic50_tensor.numpy(), np.array([1.0])
                    )

    def test_data_loader(self) -> None:
        """Test data_loader."""
        drug_sensitivity_content = os.linesep.join(
            [
                ',drug,cell_line,IC50',
                '0,CHEMBL14688,sample_3,2.1',
                '1,CHEMBL14688,sample_2,-0.9',
                '2,CHEMBL17564,sample_1,1.2',
                '3,CHEMBL17564,sample_2,1.5',
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
        gene_expression_content = os.linesep.join(
            [
                'genes,A,B,C,D',
                'sample_3,9.45,4.984,7.016,8.336',
                'sample_2,7.188,0.695,10.34,6.047',
                'sample_1,9.25,6.133,5.047,5.6',
            ]
        )
        with TestFileContent(
            drug_sensitivity_content
        ) as drug_sensitivity_file:
            with TestFileContent(smiles_content) as smiles_file:
                with TestFileContent(
                    gene_expression_content
                ) as gene_expression_file:
                    drug_sensitivity_dataset = DrugSensitivityDataset(
                        drug_sensitivity_file.filename,
                        smiles_file.filename,
                        gene_expression_file.filename,
                        backend='lazy'
                    )
                    data_loader = DataLoader(
                        drug_sensitivity_dataset, batch_size=2, shuffle=True
                    )
                    for (
                        batch_index, (
                            token_indexes_batch, gene_expression_batch,
                            ic50_batch
                        )
                    ) in enumerate(data_loader):
                        self.assertEqual(token_indexes_batch.size(), (2, 4))
                        self.assertEqual(gene_expression_batch.size(), (2, 4))
                        self.assertEqual(ic50_batch.size(), (2, 1))
                        if batch_index > 4:
                            break


if __name__ == '__main__':
    unittest.main()
