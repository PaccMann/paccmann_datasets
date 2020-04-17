"""Testing transforms."""
import unittest
from pytoda.transforms import LeftPadding


class TestTransforms(unittest.TestCase):
    """Testing transforms."""

    def test_left_padding(self) -> None:
        """Test LeftPadding."""

        padding_index = 0
        padding_lengths = [8]

        # Molecules that are too long will be cut and a warning will be raised.
        for padding_length in padding_lengths:
            transform = LeftPadding(
                padding_index=padding_index, padding_length=padding_length
            )
            for mol in ['C(N)CS', 'CCO']:
                self.assertEqual(len(transform(list(mol))), padding_length)


if __name__ == '__main__':
    unittest.main()
