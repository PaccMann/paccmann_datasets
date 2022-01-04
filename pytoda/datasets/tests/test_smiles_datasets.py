"""Testing SMILES dataset with lazy backend."""
import json
import os
import unittest

import numpy as np
from torch.utils.data import DataLoader

from pytoda.datasets import SMILESTokenizerDataset
from pytoda.smiles import SMILESTokenizer, metadata
from pytoda.smiles.smiles_language import TOKENIZER_CONFIG_FILE
from pytoda.tests.utils import TestFileContent

CONTENT = os.linesep.join(
    ['CCO	CHEMBL545', 'C	CHEMBL17564', 'CO	CHEMBL14688', 'NCCS	CHEMBL602']
)
MORE_CONTENT = os.linesep.join(
    [
        'COCC(C)N	CHEMBL3184692',
        'COCCOC	CHEMBL1232411',
        'O=CC1CCC1	CHEMBL18475',  # longest with length 9
        'NC(=O)O	CHEMBL125278',
    ]
)
LONGEST = 9


class TestSMILESTokenizerDatasetEager(unittest.TestCase):
    """Testing SMILES dataset with eager backend."""

    def setUp(self):
        self.backend = 'eager'
        print(f'backend is {self.backend}')
        self.content = CONTENT
        self.other_content = MORE_CONTENT
        self.longest = LONGEST

    def test___len__(self) -> None:
        """Test __len__."""

        with TestFileContent(self.content) as a_test_file:
            with TestFileContent(self.other_content) as another_test_file:
                smiles_dataset = SMILESTokenizerDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    backend=self.backend,
                )
                self.assertEqual(len(smiles_dataset), 8)

    def test___getitem__(self) -> None:
        """Test __getitem__."""

        with TestFileContent(self.content) as a_test_file:
            with TestFileContent(self.other_content) as another_test_file:
                smiles_dataset = SMILESTokenizerDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    padding=True,
                    augment=False,
                    kekulize=True,
                    sanitize=True,
                    all_bonds_explicit=True,
                    remove_chirality=True,
                    backend=self.backend,
                )
                pad_index = smiles_dataset.smiles_language.padding_index
                start_index = smiles_dataset.smiles_language.start_index
                stop_index = smiles_dataset.smiles_language.stop_index
                c_index = smiles_dataset.smiles_language.token_to_index['C']
                o_index = smiles_dataset.smiles_language.token_to_index['O']
                n_index = smiles_dataset.smiles_language.token_to_index['N']
                s_index = smiles_dataset.smiles_language.token_to_index['S']
                d_index = smiles_dataset.smiles_language.token_to_index['-']

                sample = 0
                padding_len = smiles_dataset.smiles_language.padding_length - (
                    len(
                        # str from underlying concatenated _smi dataset
                        smiles_dataset.dataset[sample]
                    )
                    * 2
                    - 1  # just d_index, no '=', '(', etc.
                )
                self.assertListEqual(
                    smiles_dataset[sample].numpy().flatten().tolist(),
                    [pad_index] * padding_len
                    + [c_index, d_index, c_index, d_index, o_index],
                )

                sample = 3
                padding_len = smiles_dataset.smiles_language.padding_length - (
                    len(
                        # str from underlying concatenated _smi dataset
                        smiles_dataset.dataset[sample]
                    )
                    * 2
                    - 1  # just d_index, no '=', '(', etc.
                )
                self.assertListEqual(
                    smiles_dataset[sample].numpy().flatten().tolist(),
                    [pad_index] * padding_len
                    + [n_index, d_index, c_index, d_index, c_index, d_index, s_index],
                )

                sample = 5
                padding_len = smiles_dataset.smiles_language.padding_length - (
                    len(
                        # str from underlying concatenated _smi dataset
                        smiles_dataset.dataset[sample]
                    )
                    * 2
                    - 1  # just d_index, no '=', '(', etc.
                )
                self.assertListEqual(
                    smiles_dataset[sample].numpy().flatten().tolist(),
                    [pad_index] * padding_len
                    + [
                        c_index,
                        d_index,
                        o_index,
                        d_index,
                        c_index,
                        d_index,
                        c_index,
                        d_index,
                        o_index,
                        d_index,
                        c_index,
                    ],
                )

                smiles_dataset = SMILESTokenizerDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    padding=False,
                    backend=self.backend,
                )
                c_index = smiles_dataset.smiles_language.token_to_index['C']
                o_index = smiles_dataset.smiles_language.token_to_index['O']

                self.assertListEqual(
                    smiles_dataset[0].numpy().flatten().tolist(),
                    [c_index, c_index, o_index],
                )
                self.assertListEqual(
                    smiles_dataset[5].numpy().flatten().tolist(),
                    [c_index, o_index, c_index, c_index, o_index, c_index],
                )

                smiles_dataset = SMILESTokenizerDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    add_start_and_stop=True,
                    backend=self.backend,
                )
                pad_index = smiles_dataset.smiles_language.padding_index
                start_index = smiles_dataset.smiles_language.start_index
                stop_index = smiles_dataset.smiles_language.stop_index
                c_index = smiles_dataset.smiles_language.token_to_index['C']
                o_index = smiles_dataset.smiles_language.token_to_index['O']

                self.assertEqual(
                    smiles_dataset.smiles_language.padding_length, self.longest + 2
                )

                sample = 0
                self.assertEqual(smiles_dataset.dataset[sample], 'CCO')
                padding_len = (
                    smiles_dataset.smiles_language.padding_length
                    - (
                        len(
                            # str from underlying concatenated _smi dataset
                            smiles_dataset.dataset[sample]
                        )
                    )
                    - 2
                )  # start and stop
                self.assertListEqual(
                    smiles_dataset[sample].numpy().flatten().tolist(),
                    [pad_index] * padding_len
                    + [start_index, c_index, c_index, o_index, stop_index],
                )

                sample = 5
                self.assertEqual(smiles_dataset.dataset[sample], 'COCCOC')
                padding_len = (
                    smiles_dataset.smiles_language.padding_length
                    - (
                        len(
                            # str from underlying concatenated _smi dataset
                            smiles_dataset.dataset[sample]
                        )
                    )
                    - 2
                )  # start and stop
                self.assertListEqual(
                    smiles_dataset[sample].numpy().flatten().tolist(),
                    [pad_index] * padding_len
                    + [
                        start_index,
                        c_index,
                        o_index,
                        c_index,
                        c_index,
                        o_index,
                        c_index,
                        stop_index,
                    ],
                )

                smiles_dataset = SMILESTokenizerDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    augment=True,
                    backend=self.backend,
                )
                np.random.seed(0)
                for randomized_smiles in ['C(S)CN', 'NCCS', 'SCCN', 'C(N)CS', 'C(CS)N']:
                    token_indexes = smiles_dataset[3].numpy().flatten().tolist()
                    smiles = smiles_dataset.smiles_language.token_indexes_to_smiles(
                        token_indexes
                    )
                    self.assertEqual(smiles, randomized_smiles)

                smiles_dataset = SMILESTokenizerDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    padding=False,
                    add_start_and_stop=True,
                    remove_bonddir=True,
                    selfies=True,
                    backend=self.backend,
                )
                c_index = smiles_dataset.smiles_language.token_to_index['[C]']
                o_index = smiles_dataset.smiles_language.token_to_index['[O]']
                n_index = smiles_dataset.smiles_language.token_to_index['[N]']
                s_index = smiles_dataset.smiles_language.token_to_index['[S]']

                self.assertListEqual(
                    smiles_dataset[0].numpy().flatten().tolist(),
                    [start_index, c_index, c_index, o_index, stop_index],
                )
                self.assertListEqual(
                    smiles_dataset[3].numpy().flatten().tolist(),
                    [start_index, n_index, c_index, c_index, s_index, stop_index],
                )

    def test_data_loader(self) -> None:
        """Test data_loader."""
        with TestFileContent(self.content) as a_test_file:
            with TestFileContent(self.other_content) as another_test_file:
                smiles_dataset = SMILESTokenizerDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    backend=self.backend,
                )
                data_loader = DataLoader(smiles_dataset, batch_size=4, shuffle=True)
                for batch_index, batch in enumerate(data_loader):
                    self.assertEqual(batch.shape, (4, self.longest))
                    if batch_index > 10:
                        break

    def _test_indexed(self, ds, keys, index):
        key = keys[index]
        positive_index = index % len(ds)
        # get_key (support for negative index?)
        self.assertEqual(key, ds.get_key(positive_index))
        self.assertEqual(key, ds.get_key(index))
        # get_index
        self.assertEqual(positive_index, ds.get_index(key))
        # get_item_from_key
        self.assertTrue(all(ds[index] == ds.get_item_from_key(key)))
        # keys
        self.assertSequenceEqual(keys, list(ds.keys()))
        # duplicate keys
        self.assertFalse(ds.has_duplicate_keys)

    def test_all_base_for_indexed_methods(self):

        with TestFileContent(self.content) as a_test_file:
            with TestFileContent(self.other_content) as another_test_file:
                smiles_dataset = SMILESTokenizerDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    backend=self.backend,
                )
                smiles_dataset_0 = SMILESTokenizerDataset(
                    a_test_file.filename, backend=self.backend
                )
                smiles_dataset_1 = SMILESTokenizerDataset(
                    another_test_file.filename, backend=self.backend
                )
        all_smiles, all_keys = zip(
            *(
                pair.split('\t')
                for pair in self.content.split(os.linesep)
                + self.other_content.split(os.linesep)
            )
        )

        for ds, keys in [
            (smiles_dataset, all_keys),
            (smiles_dataset_0, all_keys[:4]),
            (smiles_dataset_1, all_keys[4:]),
            # no transformation on
            # concat delegation to _SmiLazyDataset/_SmiEagerDataset
            (smiles_dataset_0 + smiles_dataset_1, all_keys),
        ]:
            index = -1
            self._test_indexed(ds, keys, index)

        # duplicate
        duplicate_ds = smiles_dataset_0 + smiles_dataset_0
        self.assertTrue(duplicate_ds.has_duplicate_keys)

        # SMILESTokenizerDataset tests and raises
        with TestFileContent(self.content) as a_test_file:
            with self.assertRaises(KeyError):
                smiles_dataset = SMILESTokenizerDataset(
                    a_test_file.filename, a_test_file.filename, backend=self.backend
                )

    def test_pretrained__getitem__(self) -> None:
        """Test __getitem__."""
        pretrained_path = os.path.join(
            os.path.dirname(os.path.abspath(metadata.__file__)),
            'tokenizer',
        )
        smiles_language = SMILESTokenizer.from_pretrained(pretrained_path, padding=True)

        with TestFileContent(self.content) as a_test_file:
            with TestFileContent(self.other_content) as another_test_file:

                smiles_dataset = SMILESTokenizerDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    smiles_language=smiles_language,
                    iterate_dataset=False,
                )

                config_file = os.path.join(pretrained_path, TOKENIZER_CONFIG_FILE)
                with open(config_file, encoding="utf-8") as fp:
                    max_length = json.load(fp)['max_token_sequence_length']
            pad_index = smiles_dataset.smiles_language.padding_index
            c_index = smiles_dataset.smiles_language.token_to_index['C']
            o_index = smiles_dataset.smiles_language.token_to_index['O']

            self.assertEqual(max_length, smiles_dataset.smiles_language.padding_length)
            sample = 0
            padding_len = smiles_dataset.smiles_language.padding_length - (
                len(
                    # str from underlying concatenated _smi dataset
                    smiles_dataset.dataset[sample]
                )
            )
            self.assertListEqual(
                smiles_dataset[sample].numpy().flatten().tolist(),
                [pad_index] * padding_len + [c_index, c_index, o_index],
            )

        # Test case where the language is adapted
        smiles_language = SMILESTokenizer.from_pretrained(
            pretrained_path, padding=False
        )

        with TestFileContent(self.content) as a_test_file:
            with TestFileContent(self.other_content) as another_test_file:

                smiles_dataset = SMILESTokenizerDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    smiles_language=smiles_language,
                    iterate_dataset=False,
                )

                config_file = os.path.join(pretrained_path, TOKENIZER_CONFIG_FILE)
                with open(config_file, encoding="utf-8") as fp:
                    max_length = json.load(fp)['max_token_sequence_length']
            c_index = smiles_dataset.smiles_language.token_to_index['C']
            o_index = smiles_dataset.smiles_language.token_to_index['O']

            sample = 0
            self.assertListEqual(
                smiles_dataset[sample].numpy().flatten().tolist(),
                [c_index, c_index, o_index],
            )

    def test_kwargs_read_smi(self):
        with TestFileContent(
            os.linesep.join(
                [
                    'CHEMBL545	metadata	CCO	and	so	on',
                    'CHEMBL17564	metadata	C	and	so	on',
                    'CHEMBL14688	metadata	CO	and	so	on',
                    'CHEMBL602	metadata	NCCS	and	so	on',
                ]
            )
        ) as test_file:
            smiles_dataset = SMILESTokenizerDataset(
                test_file.filename,
                index_col=0,
                names=['METADATA', 'SMILES', 'AND', 'SO', 'ON'],
            )
            self.assertEqual(smiles_dataset.dataset[2], 'CO')


class TestSMILESTokenizerDatasetLazy(TestSMILESTokenizerDatasetEager):
    """Testing SMILES dataset with lazy backend."""

    def setUp(self):
        self.backend = 'lazy'
        print(f'backend is {self.backend}')
        self.content = CONTENT
        self.other_content = MORE_CONTENT
        self.longest = LONGEST


if __name__ == '__main__':
    unittest.main()
