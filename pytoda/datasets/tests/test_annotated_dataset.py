"""Testing AnnotatedDataset dataset with eager backend."""
import os
import unittest

import numpy as np

from pytoda.datasets import AnnotatedDataset, SMILESTokenizerDataset, indexed, keyed
from pytoda.tests.utils import TestFileContent

# must contain all keys in annotated
SMILES_CONTENT = os.linesep.join(
    ['CCO	CHEMBL545', 'C	CHEMBL17564', 'CO	CHEMBL14688', 'NCCS	CHEMBL602']
)
ANNOTATED_CONTENT = os.linesep.join(
    [
        'label_0,label_1,annotation_index',
        '2.3,3.4,CHEMBL545',
        '4.5,5.6,CHEMBL17564',
        '6.7,7.8,CHEMBL602',
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
                smiles_dataset = SMILESTokenizerDataset(
                    smiles_file.filename, add_start_and_stop=True, backend='eager'
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
            smiles_tokens.numpy().flatten().tolist(),
            [pad_index, start_index, c_index, c_index, o_index, stop_index],
        )
        self.assertTrue(np.allclose(labels.numpy().flatten().tolist(), [2.3, 3.4]))
        # test last sample
        smiles_tokens, labels = annotated_dataset[2]
        self.assertEqual(
            smiles_tokens.numpy().flatten().tolist(),
            [start_index, n_index, c_index, c_index, s_index, stop_index],
        )
        self.assertTrue(np.allclose(labels.numpy().flatten().tolist(), [6.7, 7.8]))

    def test___getitem___from_indexed_annotation(self) -> None:
        """Test __getitem__ with index in the annotation file."""
        with TestFileContent(self.smiles_content) as smiles_file:
            with TestFileContent(self.annotated_content) as annotation_file:
                smiles_dataset = SMILESTokenizerDataset(
                    smiles_file.filename, add_start_and_stop=True, backend='eager'
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
            smiles_tokens.numpy().flatten().tolist(),
            [pad_index, start_index, c_index, c_index, o_index, stop_index],
        )
        self.assertTrue(np.allclose(labels.numpy().flatten().tolist(), [2.3, 3.4]))
        # test last sample
        smiles_tokens, labels = annotated_dataset[2]
        self.assertEqual(
            smiles_tokens.numpy().flatten().tolist(),
            [start_index, n_index, c_index, c_index, s_index, stop_index],
        )
        self.assertTrue(np.allclose(labels.numpy().flatten().tolist(), [6.7, 7.8]))

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
                smiles_dataset = SMILESTokenizerDataset(
                    smiles_file.filename, add_start_and_stop=True, backend='eager'
                )
                annotated_dataset = AnnotatedDataset(
                    annotation_file.filename,
                    dataset=smiles_dataset,
                    index_col=0,
                    label_columns=['label_1'],
                )
                duplicate_ds = AnnotatedDataset(
                    annotation_file.filename,
                    dataset=smiles_dataset + smiles_dataset,
                )

        all_keys = [
            row.split(',')[-1] for row in self.annotated_content.split(os.linesep)[1:]
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


class TestChangeIndexingReturn(unittest.TestCase):
    """Testing annotated dataset."""

    def setUp(self):
        self.smiles_content = SMILES_CONTENT
        self.annotated_content = ANNOTATED_CONTENT

    def test_return_integer_index(self) -> None:
        """Test __getitem__ with index in dataset."""
        with TestFileContent(self.smiles_content) as smiles_file:
            with TestFileContent(self.annotated_content) as annotation_file:
                smiles_dataset = SMILESTokenizerDataset(smiles_file.filename)
                # default
                default_annotated_dataset = AnnotatedDataset(
                    annotation_file.filename, dataset=smiles_dataset
                )
                # outer
                indexed_annotated_dataset = indexed(default_annotated_dataset)
                # inner
                annotated_dataset = AnnotatedDataset(
                    annotation_file.filename, dataset=indexed(smiles_dataset)
                )

        # default
        # relevant to check that `smiles_dataset` was not mutated from
        # the `indexed(smiles_dataset)` call
        smiles_tokens, labels = default_annotated_dataset[2]
        self.check_CHEMBL602(smiles_tokens, labels)

        # outer __getitem__
        (smiles_tokens, labels), sample_index = indexed_annotated_dataset[2]
        self.assertEqual(sample_index, 2)
        self.check_CHEMBL602(smiles_tokens, labels)
        # outer get_item_from_key
        (
            (smiles_tokens, labels),
            sample_index,
        ) = indexed_annotated_dataset.get_item_from_key('CHEMBL602')
        self.assertEqual(sample_index, 2)
        self.check_CHEMBL602(smiles_tokens, labels)

        # inner __getitem__
        (smiles_tokens, sample_index), labels = annotated_dataset[0]
        self.assertEqual(sample_index, 0)
        # inner __getitem__ with different index in smiles_dataset
        (smiles_tokens, sample_index), labels = annotated_dataset[2]
        self.assertEqual(sample_index, 3)
        self.check_CHEMBL602(smiles_tokens, labels)
        # inner get_item_from_key
        (smiles_tokens, sample_index), labels = annotated_dataset.get_item_from_key(
            'CHEMBL602'
        )
        self.assertEqual(sample_index, 3)
        self.check_CHEMBL602(smiles_tokens, labels)

    def test_return_key(self) -> None:
        """Test __getitem__ with key in dataset."""
        with TestFileContent(self.smiles_content) as smiles_file:
            with TestFileContent(self.annotated_content) as annotation_file:
                smiles_dataset = SMILESTokenizerDataset(smiles_file.filename)
                # default
                default_annotated_dataset = AnnotatedDataset(
                    annotation_file.filename, dataset=smiles_dataset
                )
                # outer
                keyed_annotated_dataset = keyed(default_annotated_dataset)
                # inner
                annotated_dataset = AnnotatedDataset(
                    annotation_file.filename, dataset=keyed(smiles_dataset)
                )

        # default
        # relevant to check that `smiles_dataset` was not mutated from
        # the `keyed(smiles_dataset)` call
        smiles_tokens, labels = default_annotated_dataset[2]
        self.check_CHEMBL602(smiles_tokens, labels)

        # outer __getitem__
        (smiles_tokens, labels), sample_key = keyed_annotated_dataset[2]
        self.assertEqual(sample_key, 'CHEMBL602')
        self.check_CHEMBL602(smiles_tokens, labels)
        # outer get_item_from_key
        (smiles_tokens, labels), sample_key = keyed_annotated_dataset.get_item_from_key(
            'CHEMBL602'
        )
        self.assertEqual(sample_key, 'CHEMBL602')
        self.check_CHEMBL602(smiles_tokens, labels)

        # inner __getitem__
        (smiles_tokens, sample_key), labels = annotated_dataset[0]
        self.assertEqual(sample_key, 'CHEMBL545')
        # inner __getitem__ with different index in smiles_dataset
        (smiles_tokens, sample_key), labels = annotated_dataset[2]
        self.assertEqual(sample_key, 'CHEMBL602')
        self.check_CHEMBL602(smiles_tokens, labels)
        # inner get_item_from_key
        (smiles_tokens, sample_key), labels = annotated_dataset.get_item_from_key(
            'CHEMBL602'
        )
        self.assertEqual(sample_key, 'CHEMBL602')
        self.check_CHEMBL602(smiles_tokens, labels)

    def test_return_key_index_stacked(self) -> None:
        """Test __getitem__ with key in dataset."""
        with TestFileContent(self.smiles_content) as smiles_file:
            with TestFileContent(self.annotated_content) as annotation_file:
                smiles_dataset = keyed(
                    indexed(
                        SMILESTokenizerDataset(
                            smiles_file.filename,
                        )
                    )
                )
                annotated_dataset = indexed(
                    keyed(
                        AnnotatedDataset(
                            annotation_file.filename,
                            dataset=smiles_dataset,
                        )
                    )
                )

        (smiles_tokens, smiles_index), smiles_key = smiles_dataset[3]
        self.assertEqual(smiles_key, 'CHEMBL602')
        self.assertEqual(smiles_index, 3)
        self.check_CHEMBL602(smiles_tokens)
        (smiles_tokens, smiles_index), smiles_key = smiles_dataset.get_item_from_key(
            'CHEMBL602'
        )
        self.assertEqual(smiles_key, 'CHEMBL602')
        self.assertEqual(smiles_index, 3)
        self.check_CHEMBL602(smiles_tokens)

        (
            (
                (((smiles_tokens, smiles_index), smiles_key), labels),  # inner
                annotation_key,
            ),
            annotation_index,
        ) = annotated_dataset[2]
        self.assertEqual(smiles_key, 'CHEMBL602')
        self.assertEqual(smiles_index, 3)
        self.assertEqual(annotation_index, 2)
        self.assertEqual(annotation_key, 'CHEMBL602')
        (
            (
                (((smiles_tokens, smiles_index), smiles_key), labels),  # inner
                annotation_key,
            ),
            annotation_index,
        ) = annotated_dataset.get_item_from_key('CHEMBL602')
        self.assertEqual(smiles_key, 'CHEMBL602')
        self.assertEqual(smiles_index, 3)
        self.assertEqual(annotation_index, 2)
        self.assertEqual(annotation_key, 'CHEMBL602')

    def check_CHEMBL602(self, smiles_tokens, labels=None):
        """Check indexing results lack unwanted index/key from hidden calls"""
        self.assertEqual(len(smiles_tokens.numpy().flatten().tolist()), 4)
        if labels is not None:
            self.assertEqual(len(labels.numpy().flatten().tolist()), 2)


if __name__ == '__main__':
    unittest.main()
