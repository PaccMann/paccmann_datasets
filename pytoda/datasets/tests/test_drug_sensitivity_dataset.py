"""Testing DrugSensitivityDataset."""
import os
import unittest

import numpy as np
from torch.utils.data import DataLoader

from pytoda.datasets import DrugSensitivityDataset
from pytoda.tests.utils import TestFileContent

COLUMN_NAMES = [',drug,cell_line,IC50', ',molecule,omic,label']
DRUG_SENSITIVITY_CONTENT = os.linesep.join(
    [
        '0,CHEMBL14688,sample_3,2.1',
        '1,CHEMBL14688,sample_2,-0.9',
        '2,CHEMBL17564,sample_1,1.2',
        '3,CHEMBL17564,sample_2,1.5',
    ]
)
SMILES_CONTENT = os.linesep.join(
    ['CCO	CHEMBL545', 'C	CHEMBL17564', 'CO	CHEMBL14688', 'NCCS	CHEMBL602']
)
GENE_EXPRESSION_CONTENT = os.linesep.join(
    [
        'genes,A,B,C,D',
        'sample_3,9.45,4.984,7.016,8.336',
        'sample_2,7.188,0.695,10.34,6.047',
        'sample_1,9.25,6.133,5.047,5.6',
    ]
)


class TestDrugSensitivityDatasetEagerBackend(unittest.TestCase):
    """Testing DrugSensitivityDataset with eager backend."""

    def setUp(self):
        self.backend = 'eager'
        print(f'backend is {self.backend}')
        self.smiles_content = SMILES_CONTENT
        self.gene_expression_content = GENE_EXPRESSION_CONTENT

        for column_names in COLUMN_NAMES:
            self.drug_sensitivity_content = os.linesep.join(
                [column_names, DRUG_SENSITIVITY_CONTENT]
            )

            with TestFileContent(
                self.drug_sensitivity_content
            ) as drug_sensitivity_file:
                with TestFileContent(self.smiles_content) as smiles_file:
                    with TestFileContent(
                        self.gene_expression_content
                    ) as gene_expression_file:
                        self.drug_sensitivity_dataset = DrugSensitivityDataset(
                            drug_sensitivity_file.filename,
                            smiles_file.filename,
                            gene_expression_file.filename,
                            gene_expression_kwargs={'pandas_dtype': {'genes': str}},
                            backend=self.backend,
                            column_names=column_names.split(',')[1:],
                        )

    def test___len__(self) -> None:
        """Test __len__."""
        self.assertEqual(len(self.drug_sensitivity_dataset), 4)

    def test___getitem__(self) -> None:
        """Test __getitem__."""

        padding_index = (
            self.drug_sensitivity_dataset.smiles_dataset.smiles_language.padding_index
        )
        c_index = (
            self.drug_sensitivity_dataset.smiles_dataset.smiles_language.token_to_index[
                'C'
            ]
        )
        o_index = (
            self.drug_sensitivity_dataset.smiles_dataset.smiles_language.token_to_index[
                'O'
            ]
        )
        (
            token_indexes_tensor,
            gene_expression_tensor,
            ic50_tensor,
        ) = self.drug_sensitivity_dataset[0]
        np.testing.assert_almost_equal(
            token_indexes_tensor.numpy(),
            np.array([padding_index, padding_index, c_index, o_index]),
        )
        np.testing.assert_almost_equal(
            gene_expression_tensor.numpy(),
            self.drug_sensitivity_dataset.gene_expression_dataset.get_item_from_key(
                'sample_3'
            ).numpy(),
        )
        np.testing.assert_almost_equal(ic50_tensor.numpy(), np.array([1.0]))

    def test_data_loader(self) -> None:
        """Test data_loader."""

        data_loader = DataLoader(
            self.drug_sensitivity_dataset, batch_size=2, shuffle=True
        )
        for (
            batch_index,
            (token_indexes_batch, gene_expression_batch, ic50_batch),
        ) in enumerate(data_loader):
            self.assertEqual(token_indexes_batch.size(), (2, 4))
            self.assertEqual(gene_expression_batch.size(), (2, 4))
            self.assertEqual(ic50_batch.size(), (2, 1))
            if batch_index > 4:
                break


class TestDrugSensitivityDatasetLazyBackend(
    TestDrugSensitivityDatasetEagerBackend
):  # noqa
    """Testing DrugSensitivityDataset with lazy backend."""

    def setUp(self):
        self.backend = 'lazy'
        print(f'backend is {self.backend}')
        self.smiles_content = SMILES_CONTENT
        self.gene_expression_content = GENE_EXPRESSION_CONTENT

        for column_names in COLUMN_NAMES:
            self.drug_sensitivity_content = os.linesep.join(
                [column_names, DRUG_SENSITIVITY_CONTENT]
            )
            with TestFileContent(
                self.drug_sensitivity_content
            ) as drug_sensitivity_file:
                with TestFileContent(self.smiles_content) as smiles_file:
                    with TestFileContent(
                        self.gene_expression_content
                    ) as gene_expression_file:
                        self.drug_sensitivity_dataset = DrugSensitivityDataset(
                            drug_sensitivity_file.filename,
                            smiles_file.filename,
                            gene_expression_file.filename,
                            gene_expression_kwargs={'pandas_dtype': {'genes': str}},
                            backend=self.backend,
                            column_names=column_names.split(',')[1:],
                        )


if __name__ == '__main__':
    unittest.main()
