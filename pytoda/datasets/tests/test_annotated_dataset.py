"""Testing AnnotatedDataset dataset with eager backend."""
import unittest
import os
import numpy as np
from pytoda.datasets import AnnotatedDataset, SMILESDataset
from pytoda.tests.utils import TestFileContent

# must contain all keys in annotated
SMILES_CONTENT = os.linesep.join(
    [
        'CCO	CHEMBL545',
        'C	CHEMBL17564',
        'CO	CHEMBL14688',
        'NCCS	CHEMBL602',
    ]
)
ANNOTATED_CONTENT = os.linesep.join(
    [
        'label_0,label_1,annotation_index',
        '2.3,3.4,CHEMBL545',
        '4.5,5.6,CHEMBL17564',
        '6.7,7.8,CHEMBL602'
    ]
)


class TestAnnotatedDataset(unittest.TestCase):
    """Testing annotated dataset."""

    def setUp(self):
        self.smiles_content = SMILES_CONTENT
        self.annotated_content = ANNOTATED_CONTENT

    def test___getitem__(self) -> None:
        """Test __getitem__."""
        with TestFileContent(self.smiles_content) as smiles_file:
            with TestFileContent(self.annotated_content) as annotation_file:
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
        with TestFileContent(self.smiles_content) as smiles_file:
            with TestFileContent(self.annotated_content) as annotation_file:
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

    def _test_indexed(self, ds, keys, index):
        key = keys[index]
        positive_index = index % len(ds)
        # get_key (support for negative index?)
        self.assertEqual(key, ds.get_key(positive_index))
        self.assertEqual(key, ds.get_key(index))
        # get_index
        self.assertEqual(positive_index, ds.get_index(key))
        # get_item_from_key returning labels as well
        for from_index, from_key in zip(ds[index], ds.get_item_from_key(key)):
            self.assertTrue(all(from_index == from_key))
        # keys
        self.assertSequenceEqual(keys, list(ds.keys()))
        # duplicate keys
        self.assertFalse(ds.has_duplicate_keys)

    def test_all_base_for_indexed_methods(self):

        with TestFileContent(self.smiles_content) as smiles_file:
            with TestFileContent(self.annotated_content) as annotation_file:
                smiles_dataset = SMILESDataset(
                    smiles_file.filename,
                    add_start_and_stop=True,
                    backend='eager'
                )
                annotated_dataset = AnnotatedDataset(
                    annotation_file.filename, dataset=smiles_dataset,
                    index_col=0, label_columns=['label_1']
                )
                duplicate_ds = AnnotatedDataset(
                    annotation_file.filename,
                    dataset=smiles_dataset+smiles_dataset,
                )

        all_keys = [
            row.split(',')[-1]
            for row
            in self.annotated_content.split('\n')[1:]
        ]

        for ds, keys in [
            (annotated_dataset, all_keys),
        ]:
            index = -1
            self._test_indexed(ds, keys, index)

        # duplicates in datasource can be checked directly
        self.assertTrue(duplicate_ds.datasource.has_duplicate_keys)
        # DataFrame is the dataset
        self.assertFalse(duplicate_ds.has_duplicate_keys)


if __name__ == '__main__':
    unittest.main()
