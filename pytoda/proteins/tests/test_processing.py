"""Testing ProteinFeatureLanguage."""
import os
import unittest

# from upfp import parse_fasta
from pytoda.proteins.processing import (
    AA_PROPERTIES_NUM,
    AA_FEAT,
    BLOSUM62,
    BLOSUM62_NORM,
    IUPAC_CODES,
)

# from pytoda.tests.utils import TestFileContent


class TestProcessing(unittest.TestCase):
    """Testing Processing."""

    def test__key_availability(self) -> None:
        """Test that the same keys are available in all alphabets."""
        self.assertEqual(AA_PROPERTIES_NUM.keys(), AA_FEAT.keys())
        self.assertEqual(BLOSUM62_NORM.keys(), BLOSUM62.keys())
        self.assertEqual(AA_PROPERTIES_NUM.keys(), BLOSUM62.keys())
        iupac = [x for x in IUPAC_CODES.values()] + ['<PAD>', '<START>', '<STOP>']
        self.assertEqual(iupac, list(BLOSUM62.keys()))

    def test__length_equal(self) -> None:
        for i in range(len(list(AA_PROPERTIES_NUM.values()))):
            self.assertEqual(len(list(AA_PROPERTIES_NUM.values())[i]), 6)
            self.assertEqual(len(list(AA_FEAT.values())[i]), 8)
            self.assertEqual(len(list(BLOSUM62.values())[i]), 26)
            self.assertEqual(len(list(BLOSUM62_NORM.values())[i]), 22)


if __name__ == '__main__':
    unittest.main()
