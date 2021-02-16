"""Testing AnnotatedDataset dataset with eager backend."""
import os
import unittest

import numpy as np

from pytoda.datasets import PolymerTokenizerDataset
from pytoda.tests.utils import TestFileContent


class TestPolymerTokenizerDataset(unittest.TestCase):
    """Testing annotated dataset."""

    def test___len__(self) -> None:

        content_monomer = os.linesep.join(
            ['CCO	CHEMBL545', 'C	CHEMBL17564', 'CO	CHEMBL14688', 'NCCS	CHEMBL602']
        )
        content_catalyst = os.linesep.join(
            [
                'N#CCCC1CCCC1=NNc1ccc([N+](=O)[O-])cc1	CHEMBL543',
                'CC	CHEMBL17',
                'NCCSCCCCC	CHEMBL6402',
            ]
        )

        annotated_content = os.linesep.join(
            [
                'label_0,label_1,monomer,catalyst',
                '2.3,3.4,CHEMBL545,CHEMBL17',
                '4.5,5.6,CHEMBL17564,CHEMBL6402',
                '6.7,7.8,CHEMBL602,CHEMBL6402',
                '6.7,7.8,CHEMBL54556,CHEMBL5434',
            ]
        )

        # longest = 51

        with TestFileContent(content_monomer) as a_test_file:
            with TestFileContent(content_catalyst) as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:
                    polymer_dataset = PolymerTokenizerDataset(
                        *[a_test_file.filename, another_test_file.filename],
                        annotations_filepath=annotation_file.filename,
                        entity_names=['monomer', 'cATalysT'],
                    )

                    self.assertEqual(len(polymer_dataset), 3)

    def test_smiles_params(self) -> None:

        content_monomer = os.linesep.join(
            ['CCO	CHEMBL545', 'C	CHEMBL17564', 'CO	CHEMBL14688', 'NCCS	CHEMBL602']
        )
        content_catalyst = os.linesep.join(
            [
                'N#CCCC1CCCC1=NNc1ccc([N+](=O)[O-])cc1	CHEMBL543',
                'CC	CHEMBL17',
                'NCCSCCCCC	CHEMBL6402',
            ]
        )

        annotated_content = os.linesep.join(
            [
                'label_0,label_1,monomer,catalyst',
                '2.3,3.4,CHEMBL545,CHEMBL17',
                '4.5,5.6,CHEMBL17564,CHEMBL6402',
                '6.7,7.8,CHEMBL602,CHEMBL6402',
            ]
        )

        with TestFileContent(content_monomer) as a_test_file:
            with TestFileContent(content_catalyst) as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:
                    polymer_dataset = PolymerTokenizerDataset(
                        *[a_test_file.filename, another_test_file.filename],
                        annotations_filepath=annotation_file.filename,
                        entity_names=['monomer', 'cATalysT'],
                        all_bonds_explicit=True,
                        all_hs_explicit=[True, False],
                        sanitize=[True, False],
                        padding_length=[9, None],
                    )

                    pad_ind = polymer_dataset.smiles_language.padding_index
                    monomer_start_ind = polymer_dataset.smiles_language.token_to_index[
                        '<MONOMER_START>'
                    ]
                    monomer_stop_ind = polymer_dataset.smiles_language.token_to_index[
                        '<MONOMER_STOP>'
                    ]
                    catalyst_start_ind = polymer_dataset.smiles_language.token_to_index[
                        '<CATALYST_START>'
                    ]
                    catalyst_stop_ind = polymer_dataset.smiles_language.token_to_index[
                        '<CATALYST_STOP>'
                    ]
                    ch3_ind = polymer_dataset.smiles_language.token_to_index['[CH3]']
                    oh_ind = polymer_dataset.smiles_language.token_to_index['[OH]']
                    ch2_ind = polymer_dataset.smiles_language.token_to_index['[CH2]']
                    b_ind = polymer_dataset.smiles_language.token_to_index['-']
                    c_ind = polymer_dataset.smiles_language.token_to_index['C']

                    # test first sample
                    monomer, catalyst, labels = polymer_dataset[0]

                    # CCO -> CH3CH2OH
                    self.assertEqual(
                        monomer.numpy().flatten().tolist(),
                        [
                            pad_ind,
                            pad_ind,
                            monomer_start_ind,
                            ch3_ind,
                            b_ind,
                            ch2_ind,
                            b_ind,
                            oh_ind,
                            monomer_stop_ind,
                        ],
                    )
                    self.assertEqual(
                        polymer_dataset.smiles_language.token_indexes_to_smiles(
                            monomer
                        ),
                        '[CH3]-[CH2]-[OH]',
                    )
                    # CC
                    self.assertEqual(
                        catalyst.numpy().flatten().tolist(),
                        [
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            catalyst_start_ind,
                            c_ind,
                            b_ind,
                            c_ind,
                            catalyst_stop_ind,
                        ],
                    )
                    self.assertEqual(
                        polymer_dataset.smiles_language.token_indexes_to_smiles(
                            catalyst
                        ),
                        'C-C',
                    )
                    self.assertTrue(
                        np.allclose(labels.numpy().flatten().tolist(), [2.3, 3.4])
                    )

    def test___getitem__(self) -> None:
        """Test __getitem__."""
        content_monomer = os.linesep.join(
            ['CCO	CHEMBL545', 'C	CHEMBL17564', 'CO	CHEMBL14688', 'NCCS	CHEMBL602']
        )
        content_catalyst = os.linesep.join(
            ['c1ccc([N+])cc1	CHEMBL543', 'CC	CHEMBL17', 'NCCSCCCCC	CHEMBL6402']
        )

        annotated_content = os.linesep.join(
            [
                'label_0,label_1,monomer,catalyst',
                '2.3,3.4,CHEMBL545,CHEMBL17',
                '4.5,5.6,CHEMBL17564,CHEMBL6402',
                '6.7,7.8,CHEMBL602,CHEMBL6402',
            ]
        )
        with TestFileContent(content_monomer) as a_test_file:
            with TestFileContent(content_catalyst) as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:
                    polymer_dataset = PolymerTokenizerDataset(
                        *[a_test_file.filename, another_test_file.filename],
                        annotations_filepath=annotation_file.filename,
                        entity_names=['monomer', 'cATalysT'],
                        remove_bonddir=True,
                    )

                    pad_ind = polymer_dataset.smiles_language.padding_index
                    monomer_start_ind = polymer_dataset.smiles_language.token_to_index[
                        '<MONOMER_START>'
                    ]
                    monomer_stop_ind = polymer_dataset.smiles_language.token_to_index[
                        '<MONOMER_STOP>'
                    ]
                    catalyst_start_ind = polymer_dataset.smiles_language.token_to_index[
                        '<CATALYST_START>'
                    ]
                    catalyst_stop_ind = polymer_dataset.smiles_language.token_to_index[
                        '<CATALYST_STOP>'
                    ]
                    c_ind = polymer_dataset.smiles_language.token_to_index['C']
                    o_ind = polymer_dataset.smiles_language.token_to_index['O']
                    n_ind = polymer_dataset.smiles_language.token_to_index['N']
                    s_ind = polymer_dataset.smiles_language.token_to_index['S']

                    # test first sample
                    monomer, catalyst, labels = polymer_dataset[0]

                    self.assertEqual(
                        monomer.numpy().flatten().tolist(),
                        [pad_ind] * 8
                        + [monomer_start_ind, c_ind, c_ind, o_ind, monomer_stop_ind],
                    )
                    self.assertEqual(
                        catalyst.numpy().flatten().tolist(),
                        [
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            catalyst_start_ind,
                            c_ind,
                            c_ind,
                            catalyst_stop_ind,
                        ],
                    )
                    self.assertTrue(
                        np.allclose(labels.numpy().flatten().tolist(), [2.3, 3.4])
                    )

                    monomer, catalyst, labels = polymer_dataset[2]

                    self.assertEqual(
                        monomer.numpy().flatten().tolist(),
                        [pad_ind] * 7
                        + [
                            monomer_start_ind,
                            n_ind,
                            c_ind,
                            c_ind,
                            s_ind,
                            monomer_stop_ind,
                        ],
                    )
                    self.assertEqual(
                        catalyst.numpy().flatten().tolist(),
                        [
                            pad_ind,
                            pad_ind,
                            catalyst_start_ind,
                            n_ind,
                            c_ind,
                            c_ind,
                            s_ind,
                            c_ind,
                            c_ind,
                            c_ind,
                            c_ind,
                            c_ind,
                            catalyst_stop_ind,
                        ],
                    )
                    self.assertTrue(
                        np.allclose(labels.numpy().flatten().tolist(), [6.7, 7.8])
                    )

    def test___getitem___with_annotation_column_names(self) -> None:
        """Test __getitem__ with annotations_column_names in the annotation."""
        content_monomer = os.linesep.join(
            ['CCO	CHEMBL545', 'C	CHEMBL17564', 'CO	CHEMBL14688', 'NCCS	CHEMBL602']
        )
        content_catalyst = os.linesep.join(
            ['c1ccc([N+])cc1	CHEMBL543', 'CC	CHEMBL17', 'NCCSCCCCC	CHEMBL6402']
        )
        annotated_content = os.linesep.join(
            [
                'index,label_0,label_1,monomer,catalyst',
                '0,2.3,3.4,CHEMBL545,CHEMBL6402',
                '1,4.5,5.6,CHEMBL17564,CHEMBL543',
                '1,6.7,7.8,CHEMBL602,CHEMBL17',
            ]
        )
        with TestFileContent(content_monomer) as a_test_file:
            with TestFileContent(content_catalyst) as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:
                    polymer_dataset = PolymerTokenizerDataset(
                        *[a_test_file.filename, another_test_file.filename],
                        annotations_filepath=annotation_file.filename,
                        entity_names=['monomer', 'cATalysT'],
                        annotations_column_names=['label_0'],
                        remove_bonddir=True,
                    )
                    pad_ind = polymer_dataset.smiles_language.padding_index
                    monomer_start_ind = polymer_dataset.smiles_language.token_to_index[
                        '<MONOMER_START>'
                    ]
                    monomer_stop_ind = polymer_dataset.smiles_language.token_to_index[
                        '<MONOMER_STOP>'
                    ]
                    catalyst_start_ind = polymer_dataset.smiles_language.token_to_index[
                        '<CATALYST_START>'
                    ]
                    catalyst_stop_ind = polymer_dataset.smiles_language.token_to_index[
                        '<CATALYST_STOP>'
                    ]
                    c_ind = polymer_dataset.smiles_language.token_to_index['C']
                    o_ind = polymer_dataset.smiles_language.token_to_index['O']
                    n_ind = polymer_dataset.smiles_language.token_to_index['N']
                    s_ind = polymer_dataset.smiles_language.token_to_index['S']

                    # test first sample
                    monomer, catalyst, labels = polymer_dataset[0]

                    self.assertEqual(
                        monomer.numpy().flatten().tolist(),
                        [pad_ind] * 8
                        + [monomer_start_ind, c_ind, c_ind, o_ind, monomer_stop_ind],
                    )
                    self.assertEqual(
                        catalyst.numpy().flatten().tolist(),
                        [
                            pad_ind,
                            pad_ind,
                            catalyst_start_ind,
                            n_ind,
                            c_ind,
                            c_ind,
                            s_ind,
                            c_ind,
                            c_ind,
                            c_ind,
                            c_ind,
                            c_ind,
                            catalyst_stop_ind,
                        ],
                    )
                    self.assertTrue(
                        np.allclose(labels.numpy().flatten().tolist(), [2.3])
                    )

                    monomer, catalyst, labels = polymer_dataset[2]

                    self.assertEqual(
                        monomer.numpy().flatten().tolist(),
                        [pad_ind] * 7
                        + [
                            monomer_start_ind,
                            n_ind,
                            c_ind,
                            c_ind,
                            s_ind,
                            monomer_stop_ind,
                        ],
                    )
                    self.assertEqual(
                        catalyst.numpy().flatten().tolist(),
                        [
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            pad_ind,
                            catalyst_start_ind,
                            c_ind,
                            c_ind,
                            catalyst_stop_ind,
                        ],
                    )
                    self.assertTrue(
                        np.allclose(labels.numpy().flatten().tolist(), [6.7])
                    )


if __name__ == '__main__':
    unittest.main()
