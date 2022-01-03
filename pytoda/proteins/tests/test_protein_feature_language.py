"""Testing ProteinFeatureLanguage."""
import os
import unittest

from pytoda.proteins.processing import (
    AA_FEAT,
    AA_PROPERTIES_NUM,
    BLOSUM62,
    BLOSUM62_NORM,
)
from pytoda.proteins.protein_feature_language import ProteinFeatureLanguage
from pytoda.tests.utils import TestFileContent


class TestProteinFeatureLanguage(unittest.TestCase):
    """Testing ProteinFeatureLanguage."""

    def test__update_max_token_sequence_length(self) -> None:
        """Test _update_max_token_sequence_length."""
        sequence = 'EGK'
        protein_language = ProteinFeatureLanguage(add_start_and_stop=False)
        self.assertEqual(protein_language.max_token_sequence_length, 0)
        protein_language.add_sequence(sequence)
        self.assertEqual(protein_language.max_token_sequence_length, 3)
        protein_language = ProteinFeatureLanguage(add_start_and_stop=True)
        self.assertEqual(protein_language.max_token_sequence_length, 2)
        protein_language.add_sequence(sequence)
        self.assertEqual(protein_language.max_token_sequence_length, 5)

        protein_language = ProteinFeatureLanguage(
            add_start_and_stop=False, features='blosum'
        )
        self.assertEqual(protein_language.max_token_sequence_length, 0)
        protein_language.add_sequence(sequence)
        self.assertEqual(protein_language.max_token_sequence_length, 3)
        protein_language = ProteinFeatureLanguage(add_start_and_stop=True)
        self.assertEqual(protein_language.max_token_sequence_length, 2)
        protein_language.add_sequence(sequence)
        self.assertEqual(protein_language.max_token_sequence_length, 5)

    def test_add_file(self) -> None:
        """Test add_file"""
        content = os.linesep.join(['EGK	ID3', 'S	ID1', 'FGAAV	ID2', 'NCCS	ID4'])
        with TestFileContent(content) as a_test_file:
            protein_language = ProteinFeatureLanguage()
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
            protein_language = ProteinFeatureLanguage(add_start_and_stop=False)
            protein_language.add_file(a_test_file.filename, file_type='.fasta')
            self.assertEqual(protein_language.max_token_sequence_length, 204)

    def test_sequence_to_token_indexes(self) -> None:
        """Test sequence_to_token_indexes."""
        sequence = 'CGX'
        protein_language = ProteinFeatureLanguage(add_start_and_stop=False)
        protein_language.add_sequence(sequence)
        self.assertListEqual(
            protein_language.sequence_to_token_indexes(sequence),
            [BLOSUM62['C'], BLOSUM62['G'], BLOSUM62['X']],
        )
        protein_language = ProteinFeatureLanguage(add_start_and_stop=True)
        protein_language.add_sequence(sequence)
        self.assertListEqual(
            protein_language.sequence_to_token_indexes(sequence),
            [
                BLOSUM62['<START>'],
                BLOSUM62['C'],
                BLOSUM62['G'],
                BLOSUM62['X'],
                BLOSUM62['<STOP>'],
            ],
        )
        # Other dictionary
        # Normed blosum
        protein_language = ProteinFeatureLanguage(
            add_start_and_stop=False, features='blosum_norm'
        )
        protein_language.add_sequence(sequence)
        self.assertListEqual(
            protein_language.sequence_to_token_indexes(sequence),
            [BLOSUM62_NORM['C'], BLOSUM62_NORM['G'], BLOSUM62_NORM['X']],
        )
        protein_language = ProteinFeatureLanguage(
            add_start_and_stop=True, features='blosum_norm'
        )
        protein_language.add_sequence(sequence)
        self.assertListEqual(
            protein_language.sequence_to_token_indexes(sequence),
            [
                BLOSUM62_NORM['<START>'],
                BLOSUM62_NORM['C'],
                BLOSUM62_NORM['G'],
                BLOSUM62_NORM['X'],
                BLOSUM62_NORM['<STOP>'],
            ],
        )
        protein_language = ProteinFeatureLanguage(
            add_start_and_stop=False, features='binary_features'
        )
        protein_language.add_sequence(sequence)
        self.assertListEqual(
            protein_language.sequence_to_token_indexes(sequence),
            [AA_PROPERTIES_NUM['C'], AA_PROPERTIES_NUM['G'], AA_PROPERTIES_NUM['X']],
        )
        protein_language = ProteinFeatureLanguage(
            add_start_and_stop=True, features='float_features'
        )
        protein_language.add_sequence(sequence)
        self.assertListEqual(
            protein_language.sequence_to_token_indexes(sequence),
            [
                AA_FEAT['<START>'],
                AA_FEAT['C'],
                AA_FEAT['G'],
                AA_FEAT['X'],
                AA_FEAT['<STOP>'],
            ],
        )
        # Test sequence with unknown token
        new_seq = 'CcX'
        self.assertListEqual(
            protein_language.sequence_to_token_indexes(new_seq),
            [
                AA_FEAT['<START>'],
                AA_FEAT['C'],
                AA_FEAT['<UNK>'],
                AA_FEAT['X'],
                AA_FEAT['<STOP>'],
            ],
        )

    def test_token_indexes_to_sequence(self) -> None:
        """Test token_indexes_to_sequence."""
        sequence = 'CGX'
        protein_language = ProteinFeatureLanguage()
        protein_language.add_sequence(sequence)
        token_indexes = [protein_language.token_to_index[token] for token in sequence]
        self.assertEqual(
            protein_language.token_indexes_to_sequence(token_indexes), 'CGX'
        )
        token_indexes = (
            [protein_language.token_to_index['<START>']]
            + token_indexes
            + [protein_language.token_to_index['<STOP>']]
        )
        protein_language = ProteinFeatureLanguage(add_start_and_stop=True)
        protein_language.add_sequence(sequence)
        self.assertEqual(
            protein_language.token_indexes_to_sequence(token_indexes), 'CGX'
        )

        protein_language = ProteinFeatureLanguage(features='float_features')
        protein_language.add_sequence(sequence)
        token_indexes = [protein_language.token_to_index[token] for token in sequence]
        self.assertEqual(
            protein_language.token_indexes_to_sequence(token_indexes), 'CGX'
        )
        token_indexes = (
            [protein_language.token_to_index['<START>']]
            + token_indexes
            + [protein_language.token_to_index['<STOP>']]
        )
        protein_language = ProteinFeatureLanguage(
            add_start_and_stop=True, features='float_features'
        )
        protein_language.add_sequence(sequence)
        self.assertEqual(
            protein_language.token_indexes_to_sequence(token_indexes), 'CGX'
        )
        """
        NOTE: token_indexes_to_sequence for binary_features is impossible as
        multiple aa have the same encoding.
        """
        # Test whether code throws exception.
        protein_language = ProteinFeatureLanguage(features='binary_features')
        protein_language.add_sequence(sequence)
        token_indexes = [protein_language.token_to_index[token] for token in sequence]
        self.assertRaises(
            Exception, protein_language.token_indexes_to_sequence, token_indexes
        )


if __name__ == '__main__':
    unittest.main()
