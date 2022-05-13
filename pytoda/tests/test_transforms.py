"""Testing transforms."""
import random
import unittest

import torch

from pytoda.transforms import (
    AugmentByReversing,
    Compose,
    LeftPadding,
    ListToTensor,
    ToTensor,
)


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

    def test_augment_by_reversing(self) -> None:
        """Test AugmentByReversing."""

        sequence = 'ABC'
        ground_truths = ['ABC', 'ABC', 'CBA']
        for k in range(15):
            for p, ground_truth in zip([0.0, 0.5, 1.0], ground_truths):
                random.seed(42)
                transform = AugmentByReversing(p=p)
                self.assertEqual(transform(sequence), ground_truth)

    def test_to_tensor(self) -> None:
        """Test ToTensor."""

        tokens = [2, 3, 4]
        transform = ToTensor()
        tensor = transform(tokens)
        self.assertListEqual(
            [tokens[0], tokens[1], tokens[2]], [tensor[0], tensor[1], tensor[2]]
        )
        self.assertTrue(torch.is_tensor(tensor))
        self.assertEqual(len(tensor), 3)
        self.assertEqual(len(tensor.shape), 1)
        self.assertRaises(TypeError, ToTensor, dtype=42)

    def test_list_to_tensor(self) -> None:
        """Test ListToTensor."""

        tokens = [(2, 3, 4), (2, 3, 4)]
        transform = ListToTensor()
        tensor = transform(tokens)
        self.assertEqual(len(tensor), 2)
        self.assertEqual(len(tensor.shape), 2)
        self.assertEqual(tensor.shape[-1], 3)
        self.assertListEqual(
            [tokens[0][0], tokens[0][1], tokens[0][2]],
            [tensor[0][0], tensor[0][1], tensor[0][2]],
        )
        self.assertTrue(torch.is_tensor(tensor))
        self.assertRaises(TypeError, ListToTensor, dtype=42)

    def test_compose(self) -> None:
        """Test Compose."""

        # Test equality
        c1 = Compose([ToTensor()])
        c2 = Compose([ToTensor()])
        c3 = Compose([LeftPadding(padding_length=2, padding_index=0), ToTensor()])
        c4 = Compose([ToTensor(dtype=torch.long)])
        self.assertTrue(c1 == c2)
        self.assertFalse(c1 == c3)
        self.assertFalse(c1 == c4)

        # Test repr
        self.assertIsNotNone(repr(c1))


if __name__ == '__main__':
    unittest.main()
