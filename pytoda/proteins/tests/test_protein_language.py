"""Testing ProteinLanguage."""
import os
import unittest

from pytoda.proteins.processing import (
    HUMAN_KINASE_ALIGNMENT_VOCAB,
    IUPAC_VOCAB,
    UNIREP_VOCAB,
)
from pytoda.proteins.protein_language import ProteinLanguage
from pytoda.tests.utils import TestFileContent


class TestProteinLanguage(unittest.TestCase):
    """Testing ProteinLanguage."""

    def test__update_max_token_sequence_length(self) -> None:
        """Test _update_max_token_sequence_length."""
        sequence = 'EGK'
        protein_language = ProteinLanguage(add_start_and_stop=False)
        self.assertEqual(protein_language.max_token_sequence_length, 0)
        protein_language.add_sequence(sequence)
        self.assertEqual(protein_language.max_token_sequence_length, 3)
        protein_language = ProteinLanguage(add_start_and_stop=True)
        self.assertEqual(protein_language.max_token_sequence_length, 2)
        protein_language.add_sequence(sequence)
        self.assertEqual(protein_language.max_token_sequence_length, 5)

        protein_language = ProteinLanguage(
            add_start_and_stop=False, amino_acid_dict='unirep'
        )
        self.assertEqual(protein_language.max_token_sequence_length, 0)
        protein_language.add_sequence(sequence)
        self.assertEqual(protein_language.max_token_sequence_length, 3)
        protein_language = ProteinLanguage(add_start_and_stop=True)
        self.assertEqual(protein_language.max_token_sequence_length, 2)
        protein_language.add_sequence(sequence)
        self.assertEqual(protein_language.max_token_sequence_length, 5)

    def test_add_file(self) -> None:
        """Test add_file"""
        content = os.linesep.join(['EGK	ID3', 'S	ID1', 'FGAAV	ID2', 'NCCS	ID4'])
        with TestFileContent(content) as a_test_file:
            protein_language = ProteinLanguage()
            protein_language.add_file(a_test_file.filename, index_col=1)
            self.assertEqual(protein_language.max_token_sequence_length, 7)

        # Test parsing of .fasta file
        content = r""">sp|Q6GZX0|005R_FRG3G Uncharacterized protein 005R OS=Frog virus 3 (isolate Goorha) OX=654924 GN=FV3-005R PE=4 SV=1
        MQNPLPEVMSPEHDKRTTTPMSKEANKFIRELDKKPGDLAVVSDFVKRNTGKRLPIGKRS
        NLYVRICDLSGTIYMGETFILESWEELYLPEPTKMEVLGTLESCCGIPPFPEWIVMVGED
        QCVYAYGDEEILLFAYSVKQLVEEGIQETGISYKYPDDISDVDEEVLQQDEEIQKIRKKT
        REFVDKDAQEFQDFLNSLDASLLS
        >sp|Q91G88|006L_IIV6 Putative KilA-N domain-containing protein 006L OS=Invertebrate iridescent virus 6 OX=176652 GN=IIV6-006L PE=3 SV=1
        MDSLNEVCYEQIKGTFYKGLFGDFPLIVDKKTGCFNATKLCVLGGKRFVDWNKTLRSKKL
        IQYYETRCDIKTESLLYEIKGDNNDEITKQITGTYLPKEFILDIASWISVEFYDKCNNII
        """

        with TestFileContent(content) as a_test_file:
            protein_language = ProteinLanguage(add_start_and_stop=False)
            protein_language.add_file(a_test_file.filename, file_type='.fasta')
            self.assertEqual(protein_language.max_token_sequence_length, 204)

    def test_sequence_to_token_indexes(self) -> None:
        """Test sequence_to_token_indexes."""
        sequence = 'CCO'
        protein_language = ProteinLanguage(add_start_and_stop=False)
        protein_language.add_sequence(sequence)
        self.assertListEqual(
            protein_language.sequence_to_token_indexes(sequence),
            [IUPAC_VOCAB['C'], IUPAC_VOCAB['C'], IUPAC_VOCAB['O']],
        )
        protein_language = ProteinLanguage(add_start_and_stop=True)
        protein_language.add_sequence(sequence)
        self.assertListEqual(
            protein_language.sequence_to_token_indexes(sequence),
            [
                IUPAC_VOCAB['<START>'],
                IUPAC_VOCAB['C'],
                IUPAC_VOCAB['C'],
                IUPAC_VOCAB['O'],
                IUPAC_VOCAB['<STOP>'],
            ],
        )

        # UniRep
        protein_language = ProteinLanguage(
            add_start_and_stop=False, amino_acid_dict='unirep'
        )
        protein_language.add_sequence(sequence)
        self.assertListEqual(
            protein_language.sequence_to_token_indexes(sequence),
            [UNIREP_VOCAB['C'], UNIREP_VOCAB['C'], UNIREP_VOCAB['O']],
        )
        protein_language = ProteinLanguage(
            add_start_and_stop=True, amino_acid_dict='unirep'
        )
        protein_language.add_sequence(sequence)
        self.assertListEqual(
            protein_language.sequence_to_token_indexes(sequence),
            [
                UNIREP_VOCAB['<START>'],
                UNIREP_VOCAB['C'],
                UNIREP_VOCAB['C'],
                UNIREP_VOCAB['O'],
                UNIREP_VOCAB['<STOP>'],
            ],
        )

        # human kinase alignment
        sequence = 'AC-C'
        protein_language = ProteinLanguage(
            add_start_and_stop=False, amino_acid_dict='human-kinase-alignment'
        )
        protein_language.add_sequence(sequence)
        self.assertListEqual(
            protein_language.sequence_to_token_indexes(sequence),
            [
                HUMAN_KINASE_ALIGNMENT_VOCAB['A'],
                HUMAN_KINASE_ALIGNMENT_VOCAB['C'],
                HUMAN_KINASE_ALIGNMENT_VOCAB['-'],
                HUMAN_KINASE_ALIGNMENT_VOCAB['C'],
            ],
        )
        protein_language = ProteinLanguage(
            add_start_and_stop=True, amino_acid_dict='human-kinase-alignment'
        )
        protein_language.add_sequence(sequence)
        self.assertListEqual(
            protein_language.sequence_to_token_indexes(sequence),
            [
                HUMAN_KINASE_ALIGNMENT_VOCAB['<START>'],
                HUMAN_KINASE_ALIGNMENT_VOCAB['A'],
                HUMAN_KINASE_ALIGNMENT_VOCAB['C'],
                HUMAN_KINASE_ALIGNMENT_VOCAB['-'],
                HUMAN_KINASE_ALIGNMENT_VOCAB['C'],
                HUMAN_KINASE_ALIGNMENT_VOCAB['<STOP>'],
            ],
        )

        # Test encoding an unknown character
        new_seq = 'Aq-C'
        self.assertListEqual(
            protein_language.sequence_to_token_indexes(new_seq),
            [
                HUMAN_KINASE_ALIGNMENT_VOCAB['<START>'],
                HUMAN_KINASE_ALIGNMENT_VOCAB['A'],
                HUMAN_KINASE_ALIGNMENT_VOCAB['<UNK>'],
                HUMAN_KINASE_ALIGNMENT_VOCAB['-'],
                HUMAN_KINASE_ALIGNMENT_VOCAB['C'],
                HUMAN_KINASE_ALIGNMENT_VOCAB['<STOP>'],
            ],
        )

    def test_token_indexes_to_sequence(self) -> None:
        """Test token_indexes_to_sequence."""
        sequence = 'CCO'
        protein_language = ProteinLanguage()
        protein_language.add_sequence(sequence)
        token_indexes = [protein_language.token_to_index[token] for token in sequence]
        self.assertEqual(
            protein_language.token_indexes_to_sequence(token_indexes), 'CCO'
        )
        token_indexes = (
            [protein_language.token_to_index['<START>']]
            + token_indexes
            + [protein_language.token_to_index['<STOP>']]
        )
        protein_language = ProteinLanguage(add_start_and_stop=True)
        protein_language.add_sequence(sequence)
        self.assertEqual(
            protein_language.token_indexes_to_sequence(token_indexes), 'CCO'
        )

        # UniRep
        protein_language = ProteinLanguage(amino_acid_dict='unirep')
        protein_language.add_sequence(sequence)
        token_indexes = [protein_language.token_to_index[token] for token in sequence]
        self.assertEqual(
            protein_language.token_indexes_to_sequence(token_indexes), 'CCO'
        )
        token_indexes = (
            [protein_language.token_to_index['<START>']]
            + token_indexes
            + [protein_language.token_to_index['<STOP>']]
        )
        protein_language = ProteinLanguage(
            add_start_and_stop=True, amino_acid_dict='unirep'
        )
        protein_language.add_sequence(sequence)
        self.assertEqual(
            protein_language.token_indexes_to_sequence(token_indexes), 'CCO'
        )

        # human kinase alignment
        sequence = 'AC-C'
        protein_language = ProteinLanguage(amino_acid_dict='human-kinase-alignment')
        protein_language.add_sequence(sequence)
        token_indexes = [protein_language.token_to_index[token] for token in sequence]
        self.assertEqual(
            protein_language.token_indexes_to_sequence(token_indexes), sequence
        )
        token_indexes = (
            [protein_language.token_to_index['<START>']]
            + token_indexes
            + [protein_language.token_to_index['<STOP>']]
        )
        protein_language = ProteinLanguage(
            add_start_and_stop=True, amino_acid_dict='human-kinase-alignment'
        )
        protein_language.add_sequence(sequence)
        self.assertEqual(
            protein_language.token_indexes_to_sequence(token_indexes), sequence
        )


if __name__ == '__main__':
    unittest.main()
