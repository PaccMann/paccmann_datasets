"""Testing SMILESLanguage."""
import os
import tempfile
import unittest

import torch

from pytoda.smiles.processing import split_selfies
from pytoda.smiles.smiles_language import (
    SELFIESLanguage,
    SMILESLanguage,
    SMILESTokenizer,
)
from pytoda.smiles.transforms import Selfies
from pytoda.tests.utils import TestFileContent


class TestSmilesLanguage(unittest.TestCase):
    """Testing SMILESLanguage."""

    def test__update_max_token_sequence_length(self) -> None:
        """Test _update_max_token_sequence_length."""
        smiles = 'CCO'
        smiles_language = SMILESLanguage()
        self.assertEqual(smiles_language.max_token_sequence_length, 0)
        smiles_language.add_smiles(smiles)
        self.assertEqual(smiles_language.max_token_sequence_length, 3)
        smiles_language = SMILESTokenizer(add_start_and_stop=True)
        self.assertEqual(smiles_language.max_token_sequence_length, 0)
        smiles_language.add_smiles(smiles)
        self.assertEqual(smiles_language.max_token_sequence_length, 5)

    def test__update_language_dictionaries_with_tokens(self) -> None:
        """Test _update_language_dictionaries_with_tokens."""
        smiles = 'CCO'
        smiles_language = SMILESLanguage()
        smiles_language._update_language_dictionaries_with_tokens(
            smiles_language.smiles_tokenizer(smiles)
        )
        self.assertTrue(
            'C' in smiles_language.token_to_index
            and 'O' in smiles_language.token_to_index
        )
        self.assertEqual(smiles_language.number_of_tokens, 37)

    def test_add_smis(self) -> None:
        """Test add_smis."""
        content = os.linesep.join(
            ['CCO	CHEMBL545', 'C	CHEMBL17564', 'CO	CHEMBL14688', 'NCCS	CHEMBL602']
        )
        with TestFileContent(content) as a_test_file:
            with TestFileContent(content) as another_test_file:
                smiles_language = SMILESLanguage()
                smiles_language.add_smis(
                    [a_test_file.filename, another_test_file.filename]
                )
                self.assertEqual(smiles_language.number_of_tokens, 39)

    def test_add_smi(self) -> None:
        """Test add_smi."""
        content = os.linesep.join(
            ['CCO	CHEMBL545', 'C	CHEMBL17564', 'CO	CHEMBL14688', 'NCCS	CHEMBL602']
        )
        with TestFileContent(content) as test_file:
            smiles_language = SMILESLanguage()
            smiles_language.add_smi(test_file.filename)
            self.assertEqual(smiles_language.number_of_tokens, 39)

    def test_add_smiles(self) -> None:
        """Test add_smiles."""
        smiles = 'CCO'
        smiles_language = SMILESLanguage()
        smiles_language.add_smiles(smiles)
        self.assertEqual(smiles_language.number_of_tokens, 37)

    def test_add_token(self) -> None:
        """Test add_token."""
        token = 'token'
        smiles_language = SMILESLanguage()
        smiles_language.add_token(token)
        self.assertEqual(smiles_language.number_of_tokens, 36)

    def test_multi_processing(self) -> None:
        """Test multiprocessing"""

        for start_stop in [False, True]:
            s = SMILESTokenizer(add_start_and_stop=start_stop)
            try:
                torch.multiprocessing.spawn(
                    s, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'
                )
            except AttributeError:
                # Catch this exception since it is likely caused by Pickling Error
                raise AttributeError()
            except Exception:
                # Other errors are okay since s is not callable and raises a TypeError
                # But spawn doesn't allow to catch error types, so we use bare Exception
                pass

    def test_smiles_to_token_indexes(self) -> None:
        """Test smiles_to_token_indexes."""
        smiles = 'CCO'
        smiles_language = SMILESLanguage()
        smiles_language.add_smiles(smiles)
        token_indexes = [smiles_language.token_to_index[token] for token in smiles]
        self.assertListEqual(
            smiles_language.smiles_to_token_indexes(smiles), token_indexes
        )

        # unknown token
        smiles_u = 'CCN'
        token_indexes_u = [
            smiles_language.token_to_index['C'],
            smiles_language.token_to_index['C'],
            smiles_language.unknown_index,
        ]
        self.assertListEqual(
            smiles_language.smiles_to_token_indexes(smiles_u), token_indexes_u
        )

        smiles_language = SMILESTokenizer(add_start_and_stop=True, padding=True)
        # fail with no padding_length defined
        with self.assertRaises(TypeError):
            smiles_language.smiles_to_token_indexes(smiles)

        smiles_language.add_smiles(smiles)
        smiles_language.set_max_padding()
        self.assertListEqual(
            list(smiles_language.smiles_to_token_indexes(smiles)),
            [smiles_language.start_index]
            + token_indexes
            + [smiles_language.stop_index],
        )

        # SELFIES
        smiles_language = SMILESLanguage(smiles_tokenizer=split_selfies)
        transform = Selfies()
        selfies = transform(smiles)
        self.assertEqual(selfies, smiles_language.smiles_to_selfies(smiles))
        smiles_language.add_smiles(selfies)
        token_indexes = [
            smiles_language.token_to_index[token] for token in ['[C]', '[C]', '[O]']
        ]
        self.assertListEqual(
            smiles_language.smiles_to_token_indexes(selfies), token_indexes
        )
        smiles_language = SMILESTokenizer(
            add_start_and_stop=True, smiles_tokenizer=split_selfies
        )
        smiles_language.add_smiles(selfies)
        self.assertListEqual(
            list(smiles_language.smiles_to_token_indexes(selfies)),
            [smiles_language.start_index]
            + token_indexes
            + [smiles_language.stop_index],
        )

    def test_token_indexes_to_smiles(self) -> None:
        """Test token_indexes_to_smiles."""
        smiles = 'CCO'
        smiles_language = SMILESLanguage()
        smiles_language.add_smiles(smiles)
        token_indexes = [smiles_language.token_to_index[token] for token in smiles]
        self.assertEqual(smiles_language.token_indexes_to_smiles(token_indexes), 'CCO')
        token_indexes = (
            [smiles_language.start_index] + token_indexes + [smiles_language.stop_index]
        )
        smiles_language = SMILESTokenizer(add_start_and_stop=True)
        smiles_language.add_smiles(smiles)
        self.assertEqual(smiles_language.token_indexes_to_smiles(token_indexes), 'CCO')

    def test_smiles_roundtrip(self) -> None:
        """Test smiles_to_token_indexes and token_indexes_to_smiles."""
        smiles = 'CCO'
        smiles_language = SMILESLanguage()
        smiles_language.add_smiles(smiles)

        self.assertEqual(
            smiles_language.token_indexes_to_smiles(
                smiles_language.smiles_to_token_indexes('CCO')
            ),
            'CCO',
        )

        smiles_language = SMILESTokenizer(add_start_and_stop=True)
        smiles_language.add_smiles(smiles)
        self.assertEqual(
            smiles_language.token_indexes_to_smiles(
                smiles_language.smiles_to_token_indexes('CCO')
            ),
            'CCO',
        )

    def test_smiles_to_selfies(self) -> None:
        """Test smiles_to_selfies and selfies_to_smiles"""
        smiles = 'CCO'
        smiles_language = SMILESLanguage()

        self.assertEqual(
            smiles,
            smiles_language.selfies_to_smiles(
                smiles_language.smiles_to_selfies(smiles)
            ),
        )

    def test_transform_changes(self):
        """Test update of transforms on parameter assignment"""
        smiles = 'CCO'
        smiles_language = SMILESTokenizer()
        self.assertEqual(smiles_language.max_token_sequence_length, 0)

        smiles_language.add_smiles(smiles)
        self.assertEqual(smiles_language.max_token_sequence_length, 3)

        smiles_language.add_start_and_stop = True
        smiles_language.add_smiles(smiles)  # is length function set?
        self.assertEqual(smiles_language.max_token_sequence_length, 5)
        tokens = smiles_language.smiles_to_token_indexes(smiles)

        smiles_language.padding = True
        print("\nExpected warning on setting padding length too small:")
        smiles_language.padding_length = 2  # 2 < 5
        print("\nExpected warning when actually truncating token indexes:")
        tokens_ = smiles_language.smiles_to_token_indexes(smiles)

        self.assertEqual(len(tokens_), 2)
        smiles_language.padding = False
        self.assertSequenceEqual(
            list(tokens), list(smiles_language.smiles_to_token_indexes(smiles))
        )

        smiles_language.set_encoding_transforms(padding=True, padding_length=2)
        print("\nExpected warning when actually truncating token indexes:")
        self.assertSequenceEqual(
            list(tokens_), list(smiles_language.smiles_to_token_indexes(smiles))
        )

        smiles_language.reset_initial_transforms()
        self.assertSequenceEqual(
            list(tokens), list(smiles_language.smiles_to_token_indexes(smiles))
        )

    def test_vocab_roundtrip(self):
        smiles = 'CCO'
        source_language = SMILESLanguage()
        source_language.add_smiles(smiles)
        # to test
        vocab = source_language.token_to_index
        vocab_ = source_language.index_to_token
        max_len = source_language.max_token_sequence_length
        count = source_language.token_count
        total = source_language.number_of_tokens

        # just vocab
        with tempfile.TemporaryDirectory() as tempdir:
            source_language.save_vocabulary(tempdir)

            smiles_language = SMILESTokenizer()
            smiles_language.load_vocabulary(tempdir)
        self.assertDictEqual(vocab, smiles_language.token_to_index)
        self.assertDictEqual(vocab_, smiles_language.index_to_token)

        # pretrained
        with tempfile.TemporaryDirectory() as tempdir:
            source_language.save_pretrained(tempdir)

            smiles_language = SMILESTokenizer.from_pretrained(tempdir)
        self.assertDictEqual(vocab, smiles_language.token_to_index)
        self.assertDictEqual(vocab_, smiles_language.index_to_token)

        self.assertEqual(max_len, smiles_language.max_token_sequence_length)
        self.assertDictEqual(count, smiles_language.token_count)
        self.assertEqual(total, smiles_language.number_of_tokens)


class TestSelfiesLanguage(unittest.TestCase):
    """Testing SELFIESLanguage."""

    def test_add_dataset(self):
        dataset = ['C(', 'CC)', 'CCO', 'C', 'CO', 'NCCS']
        selfies_language = SELFIESLanguage()
        print("\nExpected 'SMILES Parse Error's while iterating a dataset:")
        selfies_language.add_dataset(dataset)
        self.assertTrue((0, 'C(') in selfies_language.invalid_molecules)
        self.assertTrue((1, 'CC)') in selfies_language.invalid_molecules)


if __name__ == '__main__':
    unittest.main()
