"""Testing PolymerTokenizer."""
import os
import tempfile
import unittest

from pytoda.smiles.polymer_language import PolymerTokenizer
from pytoda.smiles.processing import split_selfies
from pytoda.smiles.transforms import Selfies
from pytoda.tests.utils import TestFileContent


class TestPolymerTokenizer(unittest.TestCase):
    """Testing PolymerTokenizer."""

    def test__update_max_token_sequence_length(self) -> None:
        """Test _update_max_token_sequence_length."""
        smiles = 'CCO'
        entities = ['Initiator', 'Monomer', 'Catalyst']
        polymer_language = PolymerTokenizer(entity_names=entities)
        self.assertEqual(polymer_language.max_token_sequence_length, 0)
        polymer_language.add_smiles(smiles)
        self.assertEqual(polymer_language.max_token_sequence_length, 5)

    def test__update_language_dictionaries_with_tokens(self) -> None:
        """Test _update_language_dictionaries_with_tokens."""
        smiles = 'CCO'
        entities = ['Initiator', 'Monomer', 'Catalyst']
        polymer_language = PolymerTokenizer(entity_names=entities)
        polymer_language._update_language_dictionaries_with_tokens(
            polymer_language.smiles_tokenizer(smiles)
        )
        self.assertTrue(
            'C' in polymer_language.token_to_index
            and 'O' in polymer_language.token_to_index
        )
        self.assertEqual(polymer_language.number_of_tokens, 43)

    def test_add_smis(self) -> None:
        """Test add_smis."""
        content = os.linesep.join(
            ['CCO	CHEMBL545', 'C	CHEMBL17564', 'CO	CHEMBL14688', 'NCCS	CHEMBL602']
        )
        with TestFileContent(content) as a_test_file:
            with TestFileContent(content) as another_test_file:
                entities = ['Initiator', 'Monomer', 'Catalyst']
                polymer_language = PolymerTokenizer(entity_names=entities)
                polymer_language.add_smis(
                    [a_test_file.filename, another_test_file.filename]
                )
                self.assertEqual(polymer_language.number_of_tokens, 45)

    def test_add_smi(self) -> None:
        """Test add_smi."""
        content = os.linesep.join(
            ['CCO	CHEMBL545', 'C	CHEMBL17564', 'CO	CHEMBL14688', 'NCCS	CHEMBL602']
        )
        with TestFileContent(content) as test_file:
            entities = ['Initiator', 'Monomer']
            polymer_language = PolymerTokenizer(entity_names=entities)
            polymer_language.add_smi(test_file.filename)
            self.assertEqual(polymer_language.number_of_tokens, 43)

    def test_add_smiles(self) -> None:
        """Test add_smiles."""
        smiles = 'CCO'
        entities = ['Initiator', 'Monomer']
        polymer_language = PolymerTokenizer(entity_names=entities)
        polymer_language.add_smiles(smiles)
        self.assertEqual(polymer_language.number_of_tokens, 41)

    def test_smiles_to_token_indexes(self) -> None:
        """Test smiles_to_token_indexes."""

        smiles = 'CCO'
        entities = ['Initiator', 'Monomer', 'Catalyst']
        polymer_language = PolymerTokenizer(entity_names=entities)
        polymer_language.add_smiles(smiles)
        token_indexes = [polymer_language.token_to_index[token] for token in smiles]
        polymer_language.update_entity('monomer')
        self.assertListEqual(
            list(polymer_language.smiles_to_token_indexes(smiles)),
            [polymer_language.token_to_index['<MONOMER_START>']]
            + token_indexes
            + [polymer_language.token_to_index['<MONOMER_STOP>']],
        )
        polymer_language.update_entity('catalyst')
        self.assertListEqual(
            list(polymer_language.smiles_to_token_indexes(smiles)),
            [polymer_language.token_to_index['<CATALYST_START>']]
            + token_indexes
            + [polymer_language.token_to_index['<CATALYST_STOP>']],
        )

        # SELFIES
        polymer_language = PolymerTokenizer(
            entity_names=entities, smiles_tokenizer=split_selfies
        )
        transform = Selfies()
        selfies = transform(smiles)
        polymer_language.add_smiles(selfies)
        token_indexes = [
            polymer_language.token_to_index[token] for token in ['[C]', '[C]', '[O]']
        ]
        polymer_language.update_entity('monomer')
        self.assertListEqual(
            list(polymer_language.smiles_to_token_indexes(selfies)),
            [polymer_language.token_to_index['<MONOMER_START>']]
            + token_indexes
            + [polymer_language.token_to_index['<MONOMER_STOP>']],
        )

    def test_token_indexes_to_smiles(self) -> None:
        """Test token_indexes_to_smiles."""
        smiles = 'CCO'
        entities = ['Initiator', 'Monomer', 'Catalyst']
        polymer_language = PolymerTokenizer(entity_names=entities)

        polymer_language.add_smiles(smiles)
        token_indexes = [polymer_language.token_to_index[token] for token in smiles]
        self.assertEqual(polymer_language.token_indexes_to_smiles(token_indexes), 'CCO')
        token_indexes = (
            [polymer_language.token_to_index['<MONOMER_START>']]
            + token_indexes
            + [polymer_language.token_to_index['<MONOMER_STOP>']]
        )
        self.assertEqual(polymer_language.token_indexes_to_smiles(token_indexes), 'CCO')

    def test_vocab_roundtrip(self):
        smiles = 'CCO'
        entities = ['Initiator', 'Monomer', 'Catalyst']
        source_language = PolymerTokenizer(entity_names=entities)
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

            polymer_language = PolymerTokenizer(entity_names=entities)
            polymer_language.load_vocabulary(tempdir)
        self.assertDictEqual(vocab, polymer_language.token_to_index)
        self.assertDictEqual(vocab_, polymer_language.index_to_token)

        # pretrained
        with tempfile.TemporaryDirectory() as tempdir:
            source_language.save_pretrained(tempdir)

            polymer_language = PolymerTokenizer.from_pretrained(tempdir)
        self.assertDictEqual(vocab, polymer_language.token_to_index)
        self.assertDictEqual(vocab_, polymer_language.index_to_token)

        self.assertEqual(max_len, polymer_language.max_token_sequence_length)
        self.assertDictEqual(count, polymer_language.token_count)
        self.assertEqual(total, polymer_language.number_of_tokens)
        self.assertEqual(entities, polymer_language.entities)


if __name__ == '__main__':
    unittest.main()
