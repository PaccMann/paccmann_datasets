"""Testing DrugSensitivityDoseDataset."""
import os
import unittest

import numpy as np
from torch.utils.data import DataLoader

from pytoda.datasets import DrugSensitivityDoseDataset
from pytoda.smiles import SMILESTokenizer
from pytoda.tests.utils import TestFileContent

COLUMN_NAMES = [',drug,cell_line,concentration,viability', ',molecule,omic,dose,label']
DRUG_SENSITIVITY_CONTENT = os.linesep.join(
    [
        '0,CHEMBL14688,sample_3,100,0.7',
        '1,CHEMBL14688,sample_2,23,0.99',
        '2,CHEMBL17564,sample_1,0.4,0.1',
        '3,CHEMBL17564,sample_2,2.3,0.8',
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


class TestDrugSensitivityDoseDataset(unittest.TestCase):
    """Testing DrugSensitivityDoseDataset with eager backend."""

    def setUp(self):

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

                    self.smiles_tokenizer = SMILESTokenizer(**{})
                    self.smiles_tokenizer.add_smi(smiles_file.filename)
                    self.smiles_tokenizer.set_max_padding()
                    self.smiles_tokenizer.set_encoding_transforms(
                        padding=True,
                        padding_length=self.smiles_tokenizer.max_token_sequence_length,
                    )
                    with TestFileContent(
                        self.gene_expression_content
                    ) as gene_expression_file:
                        for f in [np.log10, lambda x: x]:
                            self.f = f
                            self.drug_sensitivity_dataset = DrugSensitivityDoseDataset(
                                drug_sensitivity_file.filename,
                                smiles_file.filename,
                                gene_expression_file.filename,
                                smiles_language=self.smiles_tokenizer,
                                gene_expression_kwargs={'pandas_dtype': {'genes': str}},
                                column_names=column_names.split(',')[1:],
                                dose_transform=f,
                            )
                        self.test___len__()
                        self.test___getitem__()
                        # self.test_data_loader()

    def test___len__(self) -> None:
        """Test __len__."""
        self.assertEqual(len(self.drug_sensitivity_dataset), 4)

    def test___getitem__(self) -> None:
        """Test __getitem__."""

        padding_index = self.smiles_tokenizer.padding_index
        c_index = self.smiles_tokenizer.token_to_index['C']
        o_index = self.smiles_tokenizer.token_to_index['O']
        (
            token_indexes_tensor,
            gene_expression_tensor,
            concentration_tensor,
            viability_tensor,
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
        np.testing.assert_almost_equal(viability_tensor.numpy(), np.array([0.7]))
        np.testing.assert_almost_equal(
            concentration_tensor.numpy(), np.array(self.f([100]))
        )

    def test_data_loader(self) -> None:
        """Test data_loader."""

        data_loader = DataLoader(
            self.drug_sensitivity_dataset, batch_size=3, shuffle=True, drop_last=True
        )
        for (
            batch_index,
            (smiles_batch, gene_expression_batch, conc_batch, viability_batch),
        ) in enumerate(data_loader):
            self.assertEqual(smiles_batch.size(), (3, 4))
            self.assertEqual(gene_expression_batch.size(), (3, 4))
            self.assertEqual(conc_batch.size(), (3, 1))
            self.assertEqual(viability_batch.size(), (3, 1))


if __name__ == '__main__':
    unittest.main()
