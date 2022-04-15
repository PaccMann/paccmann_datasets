"""Testing ProteinProteinInteractionDataset """
import os
import unittest

import numpy as np

from pytoda.datasets import ProteinProteinInteractionDataset
from pytoda.proteins import ProteinFeatureLanguage, ProteinLanguage
from pytoda.tests.utils import TestFileContent


class TestProteinProteinInteractionDataset(unittest.TestCase):
    """Testing annotated dataset."""

    def test___len__(self) -> None:

        content_entity_1 = os.linesep.join(['CCE	ID1', 'KCPR	ID3', 'NCCS	ID2'])
        content_entity_2 = os.linesep.join(
            ['EGK	ID3', 'S	ID1', 'FGAAV	ID2', 'NCCS	ID4']
        )

        annotated_content = os.linesep.join(
            [
                'label_0,label_1,tcr,peptide',
                '2.3,3.4,ID3,ID4',
                '4.5,5.6,ID2,ID1',
                '6.7,7.8,ID1,ID2',
            ]
        )

        with TestFileContent(content_entity_1, suffix='.smi') as a_test_file:
            with TestFileContent(content_entity_2, suffix='.smi') as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:

                    for sequence_filetype in ['.smi', 'infer']:
                        ppi_dataset = ProteinProteinInteractionDataset(
                            [a_test_file.filename, another_test_file.filename],
                            ['tcr', 'peptide'],
                            annotation_file.filename,
                            sequence_filetypes=sequence_filetype,
                        )

                    self.assertEqual(len(ppi_dataset), 3)

        # Test for length if some sequences are not there
        annotated_content = os.linesep.join(
            [
                'label_0,label_1,tcr,peptide',
                '2.3,3.4,ID3,ID4',
                '4.5,5.6,ID2,ID1',
                '6.7,7.8,ID1,ID2',
                '6.7,7.8,ID7,ID2',
                '3.14,1.61,oh,no',
            ]
        )
        with TestFileContent(content_entity_1) as a_test_file:
            with TestFileContent(content_entity_2) as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:
                    ppi_dataset = ProteinProteinInteractionDataset(
                        [a_test_file.filename, another_test_file.filename],
                        ['tcr', 'peptide'],
                        annotation_file.filename,
                        sequence_filetypes='.smi',
                    )

                    self.assertEqual(len(ppi_dataset), 3)
                    # ppi_dataset.masks_df to inspect

    def test___getitem__(self) -> None:
        """Test __getitem__."""

        content_entity_1 = os.linesep.join(['CCE	ID1', 'KCPR	ID3', 'NCCS	ID2'])
        content_entity_2 = os.linesep.join(
            ['EGK	ID3', 'S	ID1', 'FGAAV	ID2', 'NCCS	ID4']
        )

        annotated_content = os.linesep.join(
            [
                'label_0,label_1,tcr,peptide',
                '2.3,3.4,ID3,ID4',
                '4.5,5.6,ID2,ID1',
                '6.7,7.8,ID1,ID2',
            ]
        )

        with TestFileContent(content_entity_1) as a_test_file:
            with TestFileContent(content_entity_2) as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:
                    ppi_dataset = ProteinProteinInteractionDataset(
                        [a_test_file.filename, another_test_file.filename],
                        ['tcr', 'peptide'],
                        annotation_file.filename,
                        sequence_filetypes='.smi',
                    )

                    # test first sample
                    tcr, peptide, label = ppi_dataset[0]

                    tok_to_idx = ppi_dataset.protein_languages[0].token_to_index

                    self.assertEqual(
                        tcr.numpy().flatten().tolist(),
                        [
                            tok_to_idx['K'],
                            tok_to_idx['C'],
                            tok_to_idx['P'],
                            tok_to_idx['R'],
                        ],
                    )
                    self.assertEqual(
                        peptide.numpy().flatten().tolist(),
                        [
                            tok_to_idx['<PAD>'],
                            tok_to_idx['N'],
                            tok_to_idx['C'],
                            tok_to_idx['C'],
                            tok_to_idx['S'],
                        ],
                    )
                    self.assertTrue(
                        np.allclose(label.numpy().flatten().tolist(), [2.3, 3.4])
                    )

        # Test for non-case-matching entity names
        with TestFileContent(content_entity_1) as a_test_file:
            with TestFileContent(content_entity_2) as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:
                    ppi_dataset = ProteinProteinInteractionDataset(
                        [a_test_file.filename, another_test_file.filename],
                        ['TCR', 'Peptide'],
                        annotation_file.filename,
                        sequence_filetypes='.smi',
                    )

                    # test first sample
                    tcr, peptide, label = ppi_dataset[0]

                    tok_to_idx = ppi_dataset.protein_languages[0].token_to_index

                    self.assertEqual(
                        tcr.numpy().flatten().tolist(),
                        [
                            tok_to_idx['K'],
                            tok_to_idx['C'],
                            tok_to_idx['P'],
                            tok_to_idx['R'],
                        ],
                    )
                    self.assertEqual(
                        peptide.numpy().flatten().tolist(),
                        [
                            tok_to_idx['<PAD>'],
                            tok_to_idx['N'],
                            tok_to_idx['C'],
                            tok_to_idx['C'],
                            tok_to_idx['S'],
                        ],
                    )
                    self.assertTrue(
                        np.allclose(label.numpy().flatten().tolist(), [2.3, 3.4])
                    )

        # Switch label columns
        annotated_content = os.linesep.join(
            [
                'label_0,label_1,peptIDE,tcR',
                '2.3,3.4,ID3,ID4',
                '4.5,5.6,ID2,ID1',
                '6.7,7.8,ID1,ID2',
            ]
        )
        content_entity_1 = os.linesep.join(['CCE	ID1', 'KCPR	ID3', 'NCCS	ID2'])
        content_entity_2 = os.linesep.join(
            ['EGK	ID3', 'S	ID1', 'FGAAV	ID2', 'NCCS	ID4']
        )

        with TestFileContent(content_entity_1) as a_test_file:
            with TestFileContent(content_entity_2) as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:
                    ppi_dataset = ProteinProteinInteractionDataset(
                        [a_test_file.filename, another_test_file.filename],
                        ['tcr', 'peptide'],
                        annotation_file.filename,
                        sequence_filetypes='.smi',
                    )
                    self.assertEqual(len(ppi_dataset), 2)

                    # test first sample
                    tcr, peptide, label = ppi_dataset[-1]

                    tok_to_idx = ppi_dataset.protein_languages[0].token_to_index
                    self.assertTrue(
                        np.allclose(label.numpy().flatten().tolist(), [6.7, 7.8])
                    )
                    self.assertEqual(
                        tcr.numpy().flatten().tolist(),
                        [
                            tok_to_idx['N'],
                            tok_to_idx['C'],
                            tok_to_idx['C'],
                            tok_to_idx['S'],
                        ],
                    )
                    self.assertEqual(
                        peptide.numpy().flatten().tolist(),
                        [
                            tok_to_idx['<PAD>'],
                            tok_to_idx['<PAD>'],
                            tok_to_idx['<PAD>'],
                            tok_to_idx['<PAD>'],
                            tok_to_idx['S'],
                        ],
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
                        annotations_column_names=['label_0'],
                    )
                    self.assertEqual(len(ppi_dataset), 2)

                    # test first sample
                    tcr, peptide, label = ppi_dataset[-1]

                    tok_to_idx = ppi_dataset.protein_languages[0].token_to_index
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
                        annotations_column_names=[1],
                    )
                    self.assertEqual(len(ppi_dataset), 2)

                    # test first sample
                    tcr, peptide, label = ppi_dataset[-1]

                    tok_to_idx = ppi_dataset.protein_languages[0].token_to_index
                    self.assertTrue(
                        np.allclose(label.numpy().flatten().tolist(), [7.8])
                    )
        # Test for giving only one protein sequence entity
        with TestFileContent(content_entity_2) as a_test_file:
            with TestFileContent(annotated_content) as annotation_file:
                ppi_dataset = ProteinProteinInteractionDataset(
                    [a_test_file.filename],
                    ['peptide'],
                    annotation_file.filename,
                    sequence_filetypes='.smi',
                    annotations_column_names=[1],
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

                        # Test passing no protein language
                        ppi_dataset = ProteinProteinInteractionDataset(
                            [
                                a_test_file.filename,
                                another_test_file.filename,
                                third_test_file.filename,
                            ],
                            entity_names=['tcr', 'peptide', 'peptide'],
                            labels_filepath=annotation_file.filename,
                            sequence_filetypes='.smi',
                            annotations_column_names=[1],
                        )
                        self.assertEqual(len(ppi_dataset), 2)

                        # test last sample
                        data_tuple = ppi_dataset[-1]
                        self.assertEqual(len(data_tuple), 4)
                        self.assertListEqual(
                            data_tuple[1].tolist(), data_tuple[2].tolist()
                        )

                        # Test passing single protein language
                        for dic in ['iupac', 'unirep', 'human-kinase-alignment']:
                            for s in [False, True]:
                                lang = ProteinLanguage(
                                    amino_acid_dict=dic, add_start_and_stop=s
                                )
                                ppi_dataset = ProteinProteinInteractionDataset(
                                    [
                                        a_test_file.filename,
                                        another_test_file.filename,
                                        third_test_file.filename,
                                    ],
                                    entity_names=['tcr', 'peptide', 'peptide'],
                                    protein_languages=lang,
                                    labels_filepath=annotation_file.filename,
                                    sequence_filetypes='.smi',
                                    annotations_column_names=[1],
                                    add_start_and_stops=s,
                                    iterate_datasets=True,
                                )
                                self.assertEqual(len(ppi_dataset), 2)

                                # test last sample
                                data_tuple = ppi_dataset[-1]
                                self.assertEqual(len(data_tuple), 4)
                                self.assertListEqual(
                                    data_tuple[1].tolist(), data_tuple[2].tolist()
                                )
                                # Tensor should be 1D
                                self.assertEqual(1, len(data_tuple[0].shape))

                        # Test passing single protein_feature_language
                        for dic in ['blosum', 'binary_features', 'float_features']:
                            for s in [False, True]:
                                lang = ProteinFeatureLanguage(
                                    features=dic, add_start_and_stop=s
                                )

                                ppi_dataset = ProteinProteinInteractionDataset(
                                    [
                                        a_test_file.filename,
                                        another_test_file.filename,
                                        third_test_file.filename,
                                    ],
                                    entity_names=['tcr', 'peptide', 'peptide'],
                                    protein_languages=lang,
                                    labels_filepath=annotation_file.filename,
                                    sequence_filetypes='.smi',
                                    annotations_column_names=[1],
                                    add_start_and_stops=s,
                                    iterate_datasets=True,
                                )
                                self.assertEqual(len(ppi_dataset), 2)

                                # test last sample
                                data_tuple = ppi_dataset[-1]
                                self.assertEqual(len(data_tuple), 4)
                                self.assertListEqual(
                                    data_tuple[1].tolist(), data_tuple[2].tolist()
                                )
                                # Tensor should be 2D
                                self.assertEqual(2, len(data_tuple[0].shape))

                        # Test passing mixture of protein and protein_feature languages
                        for dic1 in ['iupac', 'unirep', 'human-kinase-alignment']:
                            for dic2 in ['blosum', 'float_features']:

                                lang1 = ProteinLanguage(amino_acid_dict=dic1)
                                lang2 = ProteinFeatureLanguage(features=dic2)

                                ppi_dataset = ProteinProteinInteractionDataset(
                                    [
                                        a_test_file.filename,
                                        another_test_file.filename,
                                        third_test_file.filename,
                                    ],
                                    entity_names=['tcr', 'peptide', 'peptide'],
                                    protein_languages=[lang1, lang2, lang2],
                                    labels_filepath=annotation_file.filename,
                                    sequence_filetypes='.smi',
                                    annotations_column_names=[1],
                                    add_start_and_stops=True,
                                    iterate_datasets=True,
                                )
                                self.assertEqual(len(ppi_dataset), 2)

                                # test last sample
                                data_tuple = ppi_dataset[-1]
                                self.assertEqual(len(data_tuple), 4)
                                self.assertListEqual(
                                    data_tuple[1].tolist(), data_tuple[2].tolist()
                                )
                                # Tensor dimensions
                                for i, g in zip(range(len(data_tuple) - 1), (1, 2, 2)):
                                    self.assertEqual(g, len(data_tuple[i].shape))

                                # Testing alternative ordering
                                ppi_dataset = ProteinProteinInteractionDataset(
                                    [
                                        a_test_file.filename,
                                        another_test_file.filename,
                                        third_test_file.filename,
                                    ],
                                    entity_names=['tcr', 'peptide', 'peptide'],
                                    protein_languages=[lang2, lang1, lang2],
                                    labels_filepath=annotation_file.filename,
                                    sequence_filetypes='.smi',
                                    annotations_column_names=[1],
                                    add_start_and_stops=True,
                                    iterate_datasets=True,
                                )
                                self.assertEqual(len(ppi_dataset), 2)

                                # test last sample
                                data_tuple = ppi_dataset[-2]
                                self.assertEqual(len(data_tuple), 4)
                                # Test whether decoding with PL and PFL gives the same
                                feat1 = data_tuple[1].long().tolist()
                                if dic2 == 'blosum':
                                    feat0 = [
                                        tuple(x) for x in data_tuple[0].long().tolist()
                                    ]

                                    feat2 = [
                                        tuple(x) for x in data_tuple[2].long().tolist()
                                    ]
                                elif dic2 == 'float_features':
                                    feat0 = [
                                        tuple([round(xx, 2) for xx in x])
                                        for x in data_tuple[0].double().tolist()
                                    ]
                                    feat2 = [
                                        tuple([round(xx, 2) for xx in x])
                                        for x in data_tuple[2].double().tolist()
                                    ]

                                self.assertEqual(
                                    'CCE',
                                    ppi_dataset.protein_languages[
                                        0
                                    ].token_indexes_to_sequence(feat0),
                                )
                                self.assertEqual(
                                    'FGAAV',
                                    ppi_dataset.protein_languages[
                                        1
                                    ].token_indexes_to_sequence(feat1),
                                )
                                self.assertEqual(
                                    'FGAAV',
                                    ppi_dataset.protein_languages[
                                        2
                                    ].token_indexes_to_sequence(feat2),
                                )
                                # Tensor dimensions
                                for i, g in zip(range(len(data_tuple) - 1), (2, 1, 2)):
                                    self.assertEqual(g, len(data_tuple[i].shape))

        # Test for using different padding lengths
        padding_lengths = [8, 6]
        with TestFileContent(content_entity_1) as a_test_file:
            with TestFileContent(content_entity_2) as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:
                    ppi_dataset = ProteinProteinInteractionDataset(
                        [
                            a_test_file.filename,
                            another_test_file.filename,
                        ],
                        ['tcr', 'peptide'],
                        annotation_file.filename,
                        padding_lengths=padding_lengths,
                        sequence_filetypes='.smi',
                    )
                    self.assertEqual(len(ppi_dataset), 2)

                    # test last sample
                    data_tuple = ppi_dataset[-1]
                    self.assertEqual(len(data_tuple), 3)
                    for i, p in enumerate(padding_lengths):
                        self.assertEqual(len(data_tuple[i]), p)


if __name__ == '__main__':
    unittest.main()
