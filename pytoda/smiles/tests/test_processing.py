"""Testing SMILES processing."""
import unittest
from pytoda.smiles.processing import tokenize_smiles, tokenize_selfies
from pytoda.smiles.transforms import Selfies


class TestProcessing(unittest.TestCase):
    """Testing processing."""

    def test_tokenize_smiles(self) -> None:
        """Test tokenize_smiles."""
        for smiles, ground_truth in [
            ('c1cnoc1', ['c', '1', 'c', 'n', 'o', 'c', '1']),
            (
                '[O-][n+]1ccccc1S',
                ['[O-]', '[n+]', '1', 'c', 'c', 'c', 'c', 'c', '1', 'S']
            ),
            (
                'c1snnc1-c1ccccn1', [
                    'c', '1', 's', 'n', 'n', 'c', '1', '-', 'c', '1', 'c', 'c',
                    'c', 'c', 'n', '1'
                ]
            )
        ]:
            self.assertListEqual(
                tokenize_smiles(smiles, normalize=False), ground_truth
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
