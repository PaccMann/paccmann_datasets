"""Testing SMILES processing."""
import unittest
from pytoda.smiles.processing import (
    apply_normalization_dictionary, tokenize_smiles, tokenize_selfies
)
from pytoda.smiles.transforms import Selfies


class TestProcessing(unittest.TestCase):
    """Testing processing."""

    def test_apply_normalization_dictionary(self) -> None:
        """Test apply_normalization_dictionary."""
        for smiles, ground_truth in [
            ('c1cnoc1', 'C1CNOC1'),
            ('[O-][n+]1ccccc1S', '[O-][N+]1CCCCC1S'),
            ('c1snnc1-c1ccccn1', 'C1SNNC1C1CCCCN1'),
        ]:
            self.assertEqual(
                apply_normalization_dictionary(smiles), ground_truth
            )

    def test_tokenize_smiles(self) -> None:
        """Test tokenize_smiles."""
        for smiles, ground_truth in [
            ('c1cnoc1', ['C', '1', 'C', 'N', 'O', 'C', '1']),
            (
                '[O-][n+]1ccccc1S',
                ['[O-]', '[N+]', '1', 'C', 'C', 'C', 'C', 'C', '1', 'S']
            ),
            (
                'c1snnc1-c1ccccn1', [
                    'C', '1', 'S', 'N', 'N', 'C', '1', 'C', '1', 'C', 'C', 'C',
                    'C', 'N', '1'
                ]
            )
        ]:
            self.assertListEqual(
                tokenize_smiles(smiles, normalize=True), ground_truth
            )

    def test_tokenize_selfies(self) -> None:
        """Test tokenize_selfies."""
        for smiles, ground_truth in [
            (
                'c1cnoc1',
                ['[c]', '[c]', '[n]', '[o]', '[c]', '[Ring1]', '[Ring2]']
            ),
            (
                '[O-][n+]1ccccc1S', [
                    '[O-expl]', '[n+expl]', '[c]', '[c]', '[c]', '[c]', '[c]',
                    '[Ring1]', '[Branch1_1]', '[S]'
                ]
            ),
            (
                'c1snnc1-c1ccccn1', [
                    '[c]', '[s]', '[n]', '[n]', '[c]', '[Ring1]', '[Ring2]',
                    '[-c]', '[c]', '[c]', '[c]', '[c]', '[n]', '[Ring1]',
                    '[Branch1_1]'
                ]
            )
        ]:
            transform = Selfies()
            selfies = transform(smiles)
            self.assertListEqual(tokenize_selfies(selfies), ground_truth)


if __name__ == '__main__':
    unittest.main()
