"""Testing Utils."""
import unittest

from pytoda.proteins import aas_to_smiles


class TestUtils(unittest.TestCase):
    """Testing Utils."""

    def test_aas_to_smiles(self) -> None:
        """Test aas_to_smiles."""
        sequences = ['EGK', 'A']
        ground_truth = [
            'NCCCC[C@H](NC(=O)CNC(=O)[C@@H](N)CCC(=O)O)C(=O)O',
            'C[C@H](N)C(=O)O',
        ]

        for seq, gt in zip(sequences, ground_truth):
            self.assertEqual(gt, aas_to_smiles(seq))


if __name__ == '__main__':
    unittest.main()
