"""Testing SMILESLanguage."""
import unittest
import os
from pytoda.smiles.smiles_language import SMILESLanguage
from pytoda.tests.utils import TestFileContent
from pytoda.smiles.processing import tokenize_selfies
from pytoda.smiles.transforms import Selfies


class TestSmilesLanguage(unittest.TestCase):
    """Testing SMILESLanguage."""

    def test__update_max_token_sequence_length(self) -> None:
        """Test _update_max_token_sequence_length."""
        smiles = 'CCO'
        smiles_language = SMILESLanguage()
        self.assertEqual(smiles_language.max_token_sequence_length, 0)
        smiles_language.add_smiles(smiles)
        self.assertEqual(smiles_language.max_token_sequence_length, 3)
        smiles_language = SMILESLanguage(add_start_and_stop=True)
        self.assertEqual(smiles_language.max_token_sequence_length, 2)
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
            [
                'CCO	CHEMBL545',
                'C	CHEMBL17564',
                'CO	CHEMBL14688',
                'NCCS	CHEMBL602',
            ]
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
            [
                'CCO	CHEMBL545',
                'C	CHEMBL17564',
                'CO	CHEMBL14688',
                'NCCS	CHEMBL602',
            ]
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

    def test_smiles_to_token_indexes(self) -> None:
        """Test smiles_to_token_indexes."""
        smiles = 'CCO'
        smiles_language = SMILESLanguage()
        smiles_language.add_smiles(smiles)
        token_indexes = [
            smiles_language.token_to_index[token] for token in smiles
        ]
        self.assertListEqual(
            smiles_language.smiles_to_token_indexes(smiles), token_indexes
        )
        smiles_language = SMILESLanguage(add_start_and_stop=True)
        smiles_language.add_smiles(smiles)
        self.assertListEqual(
            smiles_language.smiles_to_token_indexes(smiles),
            [smiles_language.start_index] + token_indexes +
            [smiles_language.stop_index]
        )

        # SELFIES
        smiles_language = SMILESLanguage(
            smiles_tokenizer=lambda selfies: tokenize_selfies(selfies)
        )
        transform = Selfies()
        selfies = transform(smiles)
        smiles_language.add_smiles(selfies)
        token_indexes = [
            smiles_language.token_to_index[token]
            for token in ['[C]', '[C]', '[O]']
        ]
        self.assertListEqual(
            smiles_language.smiles_to_token_indexes(selfies), token_indexes
        )
        smiles_language = SMILESLanguage(
            add_start_and_stop=True,
            smiles_tokenizer=lambda selfies: tokenize_selfies(selfies)
        )
        smiles_language.add_smiles(selfies)
        self.assertListEqual(
            smiles_language.smiles_to_token_indexes(selfies),
            [smiles_language.start_index] + token_indexes +
            [smiles_language.stop_index]
        )

    def test_token_indexes_to_smiles(self) -> None:
        """Test token_indexes_to_smiles."""
        smiles = 'CCO'
        smiles_language = SMILESLanguage()
        smiles_language.add_smiles(smiles)
        token_indexes = [
            smiles_language.token_to_index[token] for token in smiles
        ]
        self.assertEqual(
            smiles_language.token_indexes_to_smiles(token_indexes), 'CCO'
        )
        token_indexes = (
            [smiles_language.start_index] + token_indexes +
            [smiles_language.stop_index]
        )
        smiles_language = SMILESLanguage(add_start_and_stop=True)
        smiles_language.add_smiles(smiles)
        self.assertEqual(
            smiles_language.token_indexes_to_smiles(token_indexes), 'CCO'
        )


if __name__ == '__main__':
    unittest.main()
