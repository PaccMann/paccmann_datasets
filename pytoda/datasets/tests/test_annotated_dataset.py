"""Testing AnnotatedDataset dataset with eager backend."""
import unittest
import os
import numpy as np
from pytoda.datasets import AnnotatedDataset, SMILESDataset
from pytoda.tests.utils import TestFileContent


class TestAnnotatedDataset(unittest.TestCase):
    """Testing annotated dataset."""

    def test___getitem__(self) -> None:
        """Test __getitem__."""
        smiles_content = os.linesep.join(
            [
                'CCO	CHEMBL545',
                'C	CHEMBL17564',
                'CO	CHEMBL14688',
                'NCCS	CHEMBL602',
            ]
        )
        annotated_content = os.linesep.join(
            [
                'label_0,label_1,annotation_index',
                '2.3,3.4,CHEMBL545',
                '4.5,5.6,CHEMBL17564',
                '6.7,7.8,CHEMBL602'
            ]
        )
        with TestFileContent(smiles_content) as smiles_file:
            with TestFileContent(annotated_content) as annotation_file:
                smiles_dataset = SMILESDataset(
                    smiles_file.filename,
                    add_start_and_stop=True,
                    backend='eager'
                )
                annotated_dataset = AnnotatedDataset(
                    annotation_file.filename, dataset=smiles_dataset
                )
                pad_index = smiles_dataset.smiles_language.padding_index
                start_index = smiles_dataset.smiles_language.start_index
                stop_index = smiles_dataset.smiles_language.stop_index
                c_index = smiles_dataset.smiles_language.token_to_index['C']
                o_index = smiles_dataset.smiles_language.token_to_index['O']
                n_index = smiles_dataset.smiles_language.token_to_index['N']
                s_index = smiles_dataset.smiles_language.token_to_index['S']
                # test first sample
                smiles_tokens, labels = annotated_dataset[0]
                self.assertEqual(
                    smiles_tokens.numpy().flatten().tolist(), [
                        pad_index, start_index, c_index, c_index, o_index,
                        stop_index
                    ]
                )
                self.assertTrue(
                    np.allclose(labels.numpy().flatten().tolist(), [2.3, 3.4])
                )
                # test last sample
                smiles_tokens, labels = annotated_dataset[2]
                self.assertEqual(
                    smiles_tokens.numpy().flatten().tolist(), [
                        start_index, n_index, c_index, c_index, s_index,
                        stop_index
                    ]
                )
                self.assertTrue(
                    np.allclose(labels.numpy().flatten().tolist(), [6.7, 7.8])
                )

    def test___getitem___with_index(self) -> None:
        """Test __getitem__ with index in the annotation file."""
        smiles_content = os.linesep.join(
            [
                'CCO	CHEMBL545',
                'C	CHEMBL17564',
                'CO	CHEMBL14688',
                'NCCS	CHEMBL602',
            ]
        )
        annotated_content = os.linesep.join(
            [
                'index,label_0,label_1,annotation_index',
                '0,2.3,3.4,CHEMBL545', '1,4.5,5.6,CHEMBL17564',
                '1,6.7,7.8,CHEMBL602'
            ]
        )
        with TestFileContent(smiles_content) as smiles_file:
            with TestFileContent(annotated_content) as annotation_file:
                smiles_dataset = SMILESDataset(
                    smiles_file.filename,
                    add_start_and_stop=True,
                    backend='eager'
                )
                annotated_dataset = AnnotatedDataset(
                    annotation_file.filename, dataset=smiles_dataset,
                    index_col=0
                )
                pad_index = smiles_dataset.smiles_language.padding_index
                start_index = smiles_dataset.smiles_language.start_index
                stop_index = smiles_dataset.smiles_language.stop_index
                c_index = smiles_dataset.smiles_language.token_to_index['C']
                o_index = smiles_dataset.smiles_language.token_to_index['O']
                n_index = smiles_dataset.smiles_language.token_to_index['N']
                s_index = smiles_dataset.smiles_language.token_to_index['S']
                # test first sample
                smiles_tokens, labels = annotated_dataset[0]
                self.assertEqual(
                    smiles_tokens.numpy().flatten().tolist(), [
                        pad_index, start_index, c_index, c_index, o_index,
                        stop_index
                    ]
                )
                self.assertTrue(
                    np.allclose(labels.numpy().flatten().tolist(), [2.3, 3.4])
                )
                # test last sample
                smiles_tokens, labels = annotated_dataset[2]
                self.assertEqual(
                    smiles_tokens.numpy().flatten().tolist(), [
                        start_index, n_index, c_index, c_index, s_index,
                        stop_index
                    ]
                )
                self.assertTrue(
                    np.allclose(labels.numpy().flatten().tolist(), [6.7, 7.8])
                )


if __name__ == '__main__':
    unittest.main()
