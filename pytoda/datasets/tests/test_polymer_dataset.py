"""Testing AnnotatedDataset dataset with eager backend."""
import unittest
from functools import wraps

import os
import numpy as np

from pytoda.datasets._polymer_dataset import \
    _PolymerDatasetNoAnnotation, _PolymerDatasetAnnotation
from pytoda.datasets import PolymerDataset
from pytoda.tests.utils import TestFileContent


def mock_input(fn):

    @wraps(fn)
    def mock_wrapper(self):
        content_monomer = os.linesep.join(
            [
                'CCO	CHEMBL545',
                'C	CHEMBL17564',
                'CO	CHEMBL14688',
                'NCCS	CHEMBL602',
            ]
        )
        content_catalyst = os.linesep.join(
            [
                'N#CCCC1CCCC1=NNc1ccc([N+](=O)[O-])cc1	CHEMBL543',
                'CC	CHEMBL17',
                'NCCSCCCCC	CHEMBL6402',
            ]
        )

        with TestFileContent(content_monomer) as a_test_file:
            with TestFileContent(content_catalyst) as another_test_file:
                fn(
                    self,
                    mock_file_1=a_test_file,
                    mock_file_2=another_test_file
                )

    return mock_wrapper


def _getitem_helper(polymer_dataset):
    pad_ind = polymer_dataset.smiles_language.padding_index
    monomer_start_ind = (
        polymer_dataset.smiles_language.token_to_index['<MONOMER_START>']
    )
    monomer_stop_ind = (
        polymer_dataset.smiles_language.token_to_index['<MONOMER_STOP>']
    )
    catalyst_start_ind = (
        polymer_dataset.smiles_language.token_to_index['<CATALYST_START>']
    )
    catalyst_stop_ind = (
        polymer_dataset.smiles_language.token_to_index['<CATALYST_STOP>']
    )
    c_ind = polymer_dataset.smiles_language.token_to_index['C']
    o_ind = polymer_dataset.smiles_language.token_to_index['O']
    n_ind = polymer_dataset.smiles_language.token_to_index['N']
    s_ind = polymer_dataset.smiles_language.token_to_index['S']
    return (
        pad_ind, monomer_start_ind, monomer_stop_ind, catalyst_start_ind,
        catalyst_stop_ind, c_ind, o_ind, n_ind, s_ind
    )


class TestPolymerDatasetAnnotation(unittest.TestCase):
    """Testing annotated dataset."""

    def test___len__(self) -> None:

        content_monomer = os.linesep.join(
            [
                'CCO	CHEMBL545',
                'C	CHEMBL17564',
                'CO	CHEMBL14688',
                'NCCS	CHEMBL602',
            ]
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
                '4.5,5.6,CHEMBL17564,CHEMBL6402',  # yapf: disable
                '6.7,7.8,CHEMBL602,CHEMBL6402',
                '6.7,7.8,CHEMBL54556,CHEMBL5434'
            ]
        )

        with TestFileContent(content_monomer) as a_test_file:
            with TestFileContent(content_catalyst) as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:
                    polymer_dataset = _PolymerDatasetAnnotation(
                        smi_filepaths=[
                            a_test_file.filename, another_test_file.filename
                        ],
                        annotations_filepath=annotation_file.filename,
                        entity_names=['monomer', 'cATalysT']
                    )

                    self.assertEqual(len(polymer_dataset), 3)

    def test_smiles_params(self) -> None:

        content_monomer = os.linesep.join(
            [
                'CCO	CHEMBL545',
                'C	CHEMBL17564',
                'CO	CHEMBL14688',
                'NCCS	CHEMBL602',
            ]
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
                '4.5,5.6,CHEMBL17564,CHEMBL6402',  # yapf: disable
                '6.7,7.8,CHEMBL602,CHEMBL6402'
            ]
        )

        with TestFileContent(content_monomer) as a_test_file:
            with TestFileContent(content_catalyst) as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:
                    polymer_dataset = _PolymerDatasetAnnotation(
                        smi_filepaths=[
                            a_test_file.filename, another_test_file.filename
                        ],
                        annotations_filepath=annotation_file.filename,
                        entity_names=['monomer', 'cATalysT'],
                        all_bonds_explicit=True,
                        all_hs_explicit=[True, False]
                    )

                    pad_ind = polymer_dataset.smiles_language.padding_index
                    monomer_start_ind = (
                        polymer_dataset.smiles_language.
                        token_to_index['<MONOMER_START>']
                    )
                    monomer_stop_ind = (
                        polymer_dataset.smiles_language.
                        token_to_index['<MONOMER_STOP>']
                    )
                    catalyst_start_ind = (
                        polymer_dataset.smiles_language.
                        token_to_index['<CATALYST_START>']
                    )
                    catalyst_stop_ind = (
                        polymer_dataset.smiles_language.
                        token_to_index['<CATALYST_STOP>']
                    )
                    ch3_ind = polymer_dataset.smiles_language.token_to_index[
                        '[CH3]']
                    oh_ind = polymer_dataset.smiles_language.token_to_index[
                        '[OH]']
                    ch2_ind = polymer_dataset.smiles_language.token_to_index[
                        '[CH2]']
                    b_ind = polymer_dataset.smiles_language.token_to_index['-']
                    c_ind = polymer_dataset.smiles_language.token_to_index['C']

                    # test first sample
                    monomer, catalyst, labels = polymer_dataset[0]

                    self.assertEqual(
                        monomer.numpy().flatten().tolist(), [
                            pad_ind, pad_ind, monomer_start_ind, ch3_ind,
                            b_ind, ch2_ind, b_ind, oh_ind, monomer_stop_ind
                        ]
                    )
                    self.assertEqual(
                        catalyst.numpy().flatten().tolist(), [
                            pad_ind, pad_ind, pad_ind, pad_ind, pad_ind,
                            pad_ind, pad_ind, pad_ind, pad_ind, pad_ind,
                            pad_ind, pad_ind, pad_ind, pad_ind, pad_ind,
                            pad_ind, pad_ind, pad_ind, pad_ind, pad_ind,
                            pad_ind, pad_ind, pad_ind, pad_ind, pad_ind,
                            pad_ind, pad_ind, pad_ind, pad_ind, pad_ind,
                            pad_ind, pad_ind, pad_ind, pad_ind, pad_ind,
                            pad_ind, pad_ind, pad_ind, pad_ind, pad_ind,
                            pad_ind, pad_ind, pad_ind, pad_ind, pad_ind,
                            pad_ind, catalyst_start_ind, c_ind, b_ind, c_ind,
                            catalyst_stop_ind
                        ]
                    )
                    self.assertTrue(
                        np.allclose(
                            labels.numpy().flatten().tolist(), [2.3, 3.4]
                        )
                    )

    def test___getitem__(self) -> None:
        """Test __getitem__."""
        content_monomer = os.linesep.join(
            [
                'CCO	CHEMBL545',
                'C	CHEMBL17564',
                'CO	CHEMBL14688',
                'NCCS	CHEMBL602',
            ]
        )
        content_catalyst = os.linesep.join(
            [
                'c1ccc([N+])cc1	CHEMBL543',
                'CC	CHEMBL17',
                'NCCSCCCCC	CHEMBL6402',
            ]
        )

        annotated_content = os.linesep.join(
            [
                'label_0,label_1,monomer,catalyst',
                '2.3,3.4,CHEMBL545,CHEMBL17',
                '4.5,5.6,CHEMBL17564,CHEMBL6402',  # yapf: disable
                '6.7,7.8,CHEMBL602,CHEMBL6402'
            ]
        )
        with TestFileContent(content_monomer) as a_test_file:
            with TestFileContent(content_catalyst) as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:
                    polymer_dataset = _PolymerDatasetAnnotation(
                        smi_filepaths=[
                            a_test_file.filename, another_test_file.filename
                        ],
                        annotations_filepath=annotation_file.filename,
                        entity_names=['monomer', 'cATalysT'],
                        remove_bonddir=True
                    )

                    pad_ind = polymer_dataset.smiles_language.padding_index
                    monomer_start_ind = (
                        polymer_dataset.smiles_language.
                        token_to_index['<MONOMER_START>']
                    )
                    monomer_stop_ind = (
                        polymer_dataset.smiles_language.
                        token_to_index['<MONOMER_STOP>']
                    )
                    catalyst_start_ind = (
                        polymer_dataset.smiles_language.
                        token_to_index['<CATALYST_START>']
                    )
                    catalyst_stop_ind = (
                        polymer_dataset.smiles_language.
                        token_to_index['<CATALYST_STOP>']
                    )
                    c_ind = polymer_dataset.smiles_language.token_to_index['C']
                    o_ind = polymer_dataset.smiles_language.token_to_index['O']
                    n_ind = polymer_dataset.smiles_language.token_to_index['N']
                    s_ind = polymer_dataset.smiles_language.token_to_index['S']

                    # test first sample
                    monomer, catalyst, labels = polymer_dataset[0]

                    self.assertEqual(
                        monomer.numpy().flatten().tolist(), [
                            pad_ind, monomer_start_ind, c_ind, c_ind, o_ind,
                            monomer_stop_ind
                        ]
                    )
                    self.assertEqual(
                        catalyst.numpy().flatten().tolist(), [
                            pad_ind, pad_ind, pad_ind, pad_ind, pad_ind,
                            pad_ind, pad_ind, pad_ind, pad_ind,
                            catalyst_start_ind, c_ind, c_ind, catalyst_stop_ind
                        ]
                    )
                    self.assertTrue(
                        np.allclose(
                            labels.numpy().flatten().tolist(), [2.3, 3.4]
                        )
                    )

                    monomer, catalyst, labels = polymer_dataset[2]

                    self.assertEqual(
                        monomer.numpy().flatten().tolist(), [
                            monomer_start_ind, n_ind, c_ind, c_ind, s_ind,
                            monomer_stop_ind
                        ]
                    )
                    self.assertEqual(
                        catalyst.numpy().flatten().tolist(), [
                            pad_ind, pad_ind, catalyst_start_ind, n_ind, c_ind,
                            c_ind, s_ind, c_ind, c_ind, c_ind, c_ind, c_ind,
                            catalyst_stop_ind
                        ]
                    )
                    self.assertTrue(
                        np.allclose(
                            labels.numpy().flatten().tolist(), [6.7, 7.8]
                        )
                    )

    def test___getitem___with_annotation_column_names(self) -> None:
        """Test __getitem__ with annotations_column_names in the annotation."""
        content_monomer = os.linesep.join(
            [
                'CCO	CHEMBL545',
                'C	CHEMBL17564',
                'CO	CHEMBL14688',
                'NCCS	CHEMBL602',
            ]
        )
        content_catalyst = os.linesep.join(
            [
                'c1ccc([N+])cc1	CHEMBL543',
                'CC	CHEMBL17',
                'NCCSCCCCC	CHEMBL6402',
            ]
        )
        annotated_content = os.linesep.join(
            [
                'index,label_0,label_1,monomer,catalyst',
                '0,2.3,3.4,CHEMBL545,CHEMBL6402',
                '1,4.5,5.6,CHEMBL17564,CHEMBL543',
                '1,6.7,7.8,CHEMBL602,CHEMBL17'
            ]
        )
        with TestFileContent(content_monomer) as a_test_file:
            with TestFileContent(content_catalyst) as another_test_file:
                with TestFileContent(annotated_content) as annotation_file:
                    polymer_dataset = _PolymerDatasetAnnotation(
                        smi_filepaths=[
                            a_test_file.filename, another_test_file.filename
                        ],
                        annotations_filepath=annotation_file.filename,
                        entity_names=['monomer', 'cATalysT'],
                        annotations_column_names=['label_0'],
                        remove_bonddir=True
                    )
                    pad_ind = polymer_dataset.smiles_language.padding_index
                    monomer_start_ind = (
                        polymer_dataset.smiles_language.
                        token_to_index['<MONOMER_START>']
                    )
                    monomer_stop_ind = (
                        polymer_dataset.smiles_language.
                        token_to_index['<MONOMER_STOP>']
                    )
                    catalyst_start_ind = (
                        polymer_dataset.smiles_language.
                        token_to_index['<CATALYST_START>']
                    )
                    catalyst_stop_ind = (
                        polymer_dataset.smiles_language.
                        token_to_index['<CATALYST_STOP>']
                    )
                    c_ind = polymer_dataset.smiles_language.token_to_index['C']
                    o_ind = polymer_dataset.smiles_language.token_to_index['O']
                    n_ind = polymer_dataset.smiles_language.token_to_index['N']
                    s_ind = polymer_dataset.smiles_language.token_to_index['S']

                    # test first sample
                    monomer, catalyst, labels = polymer_dataset[0]

                    self.assertEqual(
                        monomer.numpy().flatten().tolist(), [
                            pad_ind, monomer_start_ind, c_ind, c_ind, o_ind,
                            monomer_stop_ind
                        ]
                    )
                    self.assertEqual(
                        catalyst.numpy().flatten().tolist(), [
                            pad_ind, pad_ind, catalyst_start_ind, n_ind, c_ind,
                            c_ind, s_ind, c_ind, c_ind, c_ind, c_ind, c_ind,
                            catalyst_stop_ind
                        ]
                    )
                    self.assertTrue(
                        np.allclose(labels.numpy().flatten().tolist(), [2.3])
                    )

                    monomer, catalyst, labels = polymer_dataset[2]

                    self.assertEqual(
                        monomer.numpy().flatten().tolist(), [
                            monomer_start_ind, n_ind, c_ind, c_ind, s_ind,
                            monomer_stop_ind
                        ]
                    )
                    self.assertEqual(
                        catalyst.numpy().flatten().tolist(), [
                            pad_ind, pad_ind, pad_ind, pad_ind, pad_ind,
                            pad_ind, pad_ind, pad_ind, pad_ind,
                            catalyst_start_ind, c_ind, c_ind, catalyst_stop_ind
                        ]
                    )
                    self.assertTrue(
                        np.allclose(labels.numpy().flatten().tolist(), [6.7])
                    )


class TestPolymerDatasetNoAnnotation(unittest.TestCase):
    """Testing the non-annotated polymer dataset"""

    @mock_input
    def test___init__(self, mock_file_1, mock_file_2) -> None:
        _PolymerDatasetNoAnnotation(
            smi_filepaths=[mock_file_1.filename, mock_file_2.filename],
            entity_names=['monomer', 'cATalysT']
        )

    @mock_input
    def test___len__(self, mock_file_1, mock_file_2) -> None:

        polymer_dataset = _PolymerDatasetNoAnnotation(
            smi_filepaths=[mock_file_1.filename, mock_file_2.filename],
            entity_names=['monomer', 'cATalysT']
        )

        self.assertEqual(len(polymer_dataset), 7)

    @mock_input
    def test___getitem__(self, mock_file_1, mock_file_2) -> None:
        polymer_dataset = _PolymerDatasetNoAnnotation(
            smi_filepaths=[mock_file_1.filename, mock_file_2.filename],
            entity_names=['monomer', 'cATalysT'],
            remove_bonddir=True
        )
        (
            pad_ind, monomer_start_ind, monomer_stop_ind, catalyst_start_ind,
            catalyst_stop_ind, c_ind, o_ind, n_ind, s_ind
        ) = _getitem_helper(polymer_dataset)

        # test retrieving one sample of each entity
        monomer = polymer_dataset['monomer', 0]
        catalyst = polymer_dataset['catalyst', 1]

        self.assertEqual(
            monomer.numpy().flatten().tolist(), [
                pad_ind, monomer_start_ind, c_ind, c_ind, o_ind,
                monomer_stop_ind
            ]
        )

        self.assertEqual(
            catalyst.numpy().flatten().tolist(), [
                *(29 * [pad_ind]), catalyst_start_ind, c_ind, c_ind,
                catalyst_stop_ind
            ]
        )

    @mock_input
    def test__return_modes(self, mock_file_1, mock_file_2) -> None:
        polymer_dataset = _PolymerDatasetNoAnnotation(
            smi_filepaths=[mock_file_1.filename, mock_file_2.filename],
            entity_names=['monomer', 'cATalysT'],
            remove_bonddir=True
        )
        polymer_dataset.set_mode_smiles()
        monomer = polymer_dataset['monomer', 3]

        self.assertEqual(monomer, '<MONOMER_START>NCCS<MONOMER_STOP>')

        polymer_dataset.set_mode_tensor()
        monomer = polymer_dataset['monomer', 3]

        (
            pad_ind, monomer_start_ind, monomer_stop_ind, catalyst_start_ind,
            catalyst_stop_ind, c_ind, o_ind, n_ind, s_ind
        ) = _getitem_helper(polymer_dataset)

        self.assertEqual(
            monomer.numpy().flatten().tolist(), [
                monomer_start_ind, n_ind, c_ind, c_ind, s_ind,
                monomer_stop_ind
            ]
        )


class TestPolymerDataset(unittest.TestCase):

    @mock_input
    def test___init__annotation(self, mock_file_1, mock_file_2) -> None:
        annotated_content = os.linesep.join(
            [
                'label_0,label_1,monomer,catalyst',
                '2.3,3.4,CHEMBL545,CHEMBL17',
                '4.5,5.6,CHEMBL17564,CHEMBL6402',  # yapf: disable
                '6.7,7.8,CHEMBL602,CHEMBL6402',
                '6.7,7.8,CHEMBL54556,CHEMBL5434'
            ]
        )
        with TestFileContent(annotated_content) as annotation_file:
            PolymerDataset(
                smi_filepaths=[mock_file_1.filename, mock_file_2.filename],
                entity_names=['monomer', 'cATalysT'],
                annotations_filepath=annotation_file.filename
            )

    @mock_input
    def test___init__no_annotation(self, mock_file_1, mock_file_2) -> None:
        PolymerDataset(
            smi_filepaths=[mock_file_1.filename, mock_file_2.filename],
            entity_names=['monomer', 'cATalysT'],
            annotations_filepath=None
        )


if __name__ == '__main__':
    unittest.main()
