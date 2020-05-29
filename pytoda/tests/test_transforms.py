"""Testing transforms."""
import unittest
from pytoda.transforms import LeftPadding, ToTensor, ListToTensor
import torch


class TestTransforms(unittest.TestCase):
    """Testing transforms."""

    def test_left_padding(self) -> None:
        """Test LeftPadding."""

        padding_index = 0
        padding_lengths = [8, 4]

        # Molecules that are too long will be cut and a warning will be raised.
        for padding_length in padding_lengths:
            transform = LeftPadding(
                padding_index=padding_index, padding_length=padding_length
            )
            for mol in ['C(N)CS', 'CCO']:
                self.assertEqual(len(transform(list(mol))), padding_length)

    def test_to_tensor(self) -> None:
        """Test ToTensor."""

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        tokens = [2, 3, 4]
        transform = ToTensor(device=device)
        tensor = transform(tokens)
        self.assertListEqual(
            [tokens[0], tokens[1], tokens[2]],
            [tensor[0], tensor[1], tensor[2]]
        )
        self.assertTrue(torch.is_tensor(tensor))
        self.assertEqual(len(tensor), 3)
        self.assertEqual(len(tensor.shape), 1)

    def test_list_to_tensor(self) -> None:
        """Test ListToTensor."""

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        tokens = [(2, 3, 4), (2, 3, 4)]
        transform = ListToTensor(device=device)
        tensor = transform(tokens)
        self.assertEqual(len(tensor), 2)
        self.assertEqual(len(tensor.shape), 2)
        self.assertEqual(tensor.shape[-1], 3)
        self.assertListEqual(
            [tokens[0][0], tokens[0][1], tokens[0][2]],
            [tensor[0][0], tensor[0][1], tensor[0][2]]
        )
        self.assertTrue(torch.is_tensor(tensor))


if __name__ == '__main__':
    unittest.main()
