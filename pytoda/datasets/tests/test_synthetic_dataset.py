import unittest
import torch
from pytoda.datasets import SyntheticDataset

distribution_type = "normal"
distribution_args = {"loc": 0, "scale": 1.0}
data_dim = 3
dataset_size = 1
set_length = 5
seed = 42


class TestSyntheticDataset(unittest.TestCase):

    def setUp(self):

        self.synthetic_dataset = SyntheticDataset(
            seed, distribution_type, distribution_args, data_dim, dataset_size
        )

    def test__len__(self) -> None:
        """Test __len__."""

        self.assertEqual(
            len(self.synthetic_dataset.__getitem__(set_length)), 1
        )

    def test__getitem__(self) -> None:
        """Test __getitem__."""

        sample1 = self.synthetic_dataset.__getitem__(set_length)
        sample2 = self.synthetic_dataset.__getitem__(set_length)

        self.assertEqual(sample1.size(1), 5)
        self.assertFalse(torch.equal(sample1, sample2))
