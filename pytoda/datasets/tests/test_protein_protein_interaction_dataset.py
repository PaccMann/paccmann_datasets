"""Testing ProteinProteinInteractionDataset """
import os
import unittest

import numpy as np

from pytoda.datasets import ProteinProteinInteractionDataset
from pytoda.tests.utils import TestFileContent


class TestProteinProteinInteractionDataset(unittest.TestCase):
    """Testing annotated dataset."""

    def test___len__(self) -> None:

        content_entity_1 = os.linesep.join(
            [
                'CCO	ID1',
                'KCPR	ID3',
                'NCCS	ID2',
            ]
        )
        content_entity_2 = os.linesep.join(
            [
                'EGK	ID3',
                'S	ID1',
                'FGAAV	ID2',
                'NCCS	ID4',
            ]
        )

        annotated_content = os.linesep.join(
            [
                'label_0,label_1,tcr,peptide',
                '2.3,3.4,ID3,ID4',
                '4.5,5.6,ID2,ID1',  # yapf: disable
                '6.7,7.8,ID1,ID2'
            ]
        )

        with TestFileContent(content_entity_1) as a_test_file:
            with TestFileContent(content_entity_2) as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:
                    ppi_dataset = ProteinProteinInteractionDataset(
                        [a_test_file.filename, another_test_file.filename],
                        ['tcr', 'peptide'],
                        annotation_file.filename,
                        sequence_filetypes='.smi'
                    )

                    self.assertEqual(len(ppi_dataset), 3)

        # Test for length if some sequences are not there
        annotated_content = os.linesep.join(
            [
                'label_0,label_1,tcr,peptide',
                '2.3,3.4,ID3,ID4',
                '4.5,5.6,ID2,ID1',  # yapf: disable
                '6.7,7.8,ID1,ID2',
                '6.7,7.8,ID7,ID2'
            ]
        )
        with TestFileContent(content_entity_1) as a_test_file:
            with TestFileContent(content_entity_2) as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:
                    ppi_dataset = ProteinProteinInteractionDataset(
                        [a_test_file.filename, another_test_file.filename],
                        ['tcr', 'peptide'],
                        annotation_file.filename,
                        sequence_filetypes='.smi'
                    )

                    self.assertEqual(len(ppi_dataset), 3)

    def test___getitem__(self) -> None:
        """Test __getitem__."""

        content_entity_1 = os.linesep.join(
            [
                'CCO	ID1',
                'KCPR	ID3',
                'NCCS	ID2',
            ]
        )
        content_entity_2 = os.linesep.join(
            [
                'EGK	ID3',
                'S	ID1',
                'FGAAV	ID2',
                'NCCS	ID4',
            ]
        )

        annotated_content = os.linesep.join(
            [
                'label_0,label_1,tcr,peptide',
                '2.3,3.4,ID3,ID4',
                '4.5,5.6,ID2,ID1',  # yapf: disable
                '6.7,7.8,ID1,ID2'
            ]
        )

        with TestFileContent(content_entity_1) as a_test_file:
            with TestFileContent(content_entity_2) as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:
                    ppi_dataset = ProteinProteinInteractionDataset(
                        [a_test_file.filename, another_test_file.filename],
                        ['tcr', 'peptide'],
                        annotation_file.filename,
                        sequence_filetypes='.smi'
                    )

                    # test first sample
                    tcr, peptide, label = ppi_dataset[0]

                    tok_to_idx = ppi_dataset.protein_language.token_to_index

                    self.assertEqual(
                        tcr.numpy().flatten().tolist(), [
                            tok_to_idx['K'], tok_to_idx['C'], tok_to_idx['P'],
                            tok_to_idx['R']
                        ]
                    )
                    self.assertEqual(
                        peptide.numpy().flatten().tolist(), [
                            tok_to_idx['<PAD>'], tok_to_idx['N'],
                            tok_to_idx['C'], tok_to_idx['C'], tok_to_idx['S']
                        ]
                    )
                    self.assertTrue(
                        np.allclose(
                            label.numpy().flatten().tolist(), [2.3, 3.4]
                        )
                    )

        # Test for non-case-matching entity names
        with TestFileContent(content_entity_1) as a_test_file:
            with TestFileContent(content_entity_2) as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:
                    ppi_dataset = ProteinProteinInteractionDataset(
                        [a_test_file.filename, another_test_file.filename],
                        ['TCR', 'Peptide'],
                        annotation_file.filename,
                        sequence_filetypes='.smi'
                    )

                    # test first sample
                    tcr, peptide, label = ppi_dataset[0]

                    tok_to_idx = ppi_dataset.protein_language.token_to_index

                    self.assertEqual(
                        tcr.numpy().flatten().tolist(), [
                            tok_to_idx['K'], tok_to_idx['C'], tok_to_idx['P'],
                            tok_to_idx['R']
                        ]
                    )
                    self.assertEqual(
                        peptide.numpy().flatten().tolist(), [
                            tok_to_idx['<PAD>'], tok_to_idx['N'],
                            tok_to_idx['C'], tok_to_idx['C'], tok_to_idx['S']
                        ]
                    )
                    self.assertTrue(
                        np.allclose(
                            label.numpy().flatten().tolist(), [2.3, 3.4]
                        )
                    )

        # Switch label columns
        annotated_content = os.linesep.join(
            [
                'label_0,label_1,peptIDE,tcR',
                '2.3,3.4,ID3,ID4',
                '4.5,5.6,ID2,ID1',  # yapf: disable
                '6.7,7.8,ID1,ID2'
            ]
        )
        content_entity_1 = os.linesep.join(
            [
                'CCO	ID1',
                'KCPR	ID3',
                'NCCS	ID2',
            ]
        )
        content_entity_2 = os.linesep.join(
            [
                'EGK	ID3',
                'S	ID1',
                'FGAAV	ID2',
                'NCCS	ID4',
            ]
        )

        with TestFileContent(content_entity_1) as a_test_file:
            with TestFileContent(content_entity_2) as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:
                    ppi_dataset = ProteinProteinInteractionDataset(
                        [a_test_file.filename, another_test_file.filename],
                        ['tcr', 'peptide'],
                        annotation_file.filename,
                        sequence_filetypes='.smi'
                    )
                    self.assertEqual(len(ppi_dataset), 2)

                    # test first sample
                    tcr, peptide, label = ppi_dataset[-1]

                    tok_to_idx = ppi_dataset.protein_language.token_to_index
                    self.assertTrue(
                        np.allclose(
                            label.numpy().flatten().tolist(), [6.7, 7.8]
                        )
                    )
                    self.assertEqual(
                        tcr.numpy().flatten().tolist(), [
                            tok_to_idx['N'], tok_to_idx['C'], tok_to_idx['C'],
                            tok_to_idx['S']
                        ]
                    )
                    self.assertEqual(
                        peptide.numpy().flatten().tolist(), [
                            tok_to_idx['<PAD>'], tok_to_idx['<PAD>'],
                            tok_to_idx['<PAD>'], tok_to_idx['<PAD>'],
                            tok_to_idx['S']
                        ]
                    )

        # Only one annotation column
        with TestFileContent(content_entity_1) as a_test_file:
            with TestFileContent(content_entity_2) as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:
                    ppi_dataset = ProteinProteinInteractionDataset(
                        [a_test_file.filename, another_test_file.filename],
                        ['tcr', 'peptide'],
                        annotation_file.filename,
                        sequence_filetypes='.smi',
                        annotations_column_names=['label_0']
                    )
                    self.assertEqual(len(ppi_dataset), 2)

                    # test first sample
                    tcr, peptide, label = ppi_dataset[-1]

                    tok_to_idx = ppi_dataset.protein_language.token_to_index
                    self.assertTrue(
                        np.allclose(label.numpy().flatten().tolist(), [6.7])
                    )

        # Annotation colum given as index
        with TestFileContent(content_entity_1) as a_test_file:
            with TestFileContent(content_entity_2) as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:
                    ppi_dataset = ProteinProteinInteractionDataset(
                        [a_test_file.filename, another_test_file.filename],
                        ['tcr', 'peptide'],
                        annotation_file.filename,
                        sequence_filetypes='.smi',
                        annotations_column_names=[1]
                    )
                    self.assertEqual(len(ppi_dataset), 2)

                    # test first sample
                    tcr, peptide, label = ppi_dataset[-1]

                    tok_to_idx = ppi_dataset.protein_language.token_to_index
                    self.assertTrue(
                        np.allclose(label.numpy().flatten().tolist(), [7.8])
                    )
        # Test for giving only one protein sequence entity
        with TestFileContent(content_entity_2) as a_test_file:
            with TestFileContent(annotated_content) as annotation_file:
                ppi_dataset = ProteinProteinInteractionDataset(
                    [a_test_file.filename], ['peptide'],
                    annotation_file.filename,
                    sequence_filetypes='.smi',
                    annotations_column_names=[1]
                )
                self.assertEqual(len(ppi_dataset), 3)

                # test last sample
                data_tuple = ppi_dataset[-1]
                self.assertEqual(len(data_tuple), 2)

        # Test for giving three entities
        with TestFileContent(content_entity_1) as a_test_file:
            with TestFileContent(content_entity_2) as another_test_file:
                with TestFileContent(content_entity_2) as third_test_file:
                    with TestFileContent(annotated_content) as annotation_file:
                        ppi_dataset = ProteinProteinInteractionDataset(
                            [
                                a_test_file.filename,
                                another_test_file.filename,
                                third_test_file.filename
                            ], ['tcr', 'peptide', 'peptide'],
                            annotation_file.filename,
                            sequence_filetypes='.smi',
                            annotations_column_names=[1]
                        )
                        self.assertEqual(len(ppi_dataset), 2)

                        # test last sample
                        data_tuple = ppi_dataset[-1]
                        self.assertEqual(len(data_tuple), 4)
                        self.assertListEqual(
                            data_tuple[1].tolist(), data_tuple[2].tolist()
                        )


if __name__ == '__main__':
    unittest.main()
