"""Testing SyntheticDataset."""
import unittest
import torch
from pytoda.datasets import SyntheticDataset

distribution_types = ['normal', 'uniform']
distribution_args = [{'loc': 0.0, 'scale': 1.0}, {'low': 0.0, 'high': 1.0}]
data_dim = 16
dataset_sizes = [100, 10000]
data_depths = [1, 10]
seeds = [-1, 1, 42]


class TestSyntheticDataset(unittest.TestCase):
    """Test SyntheticDataset class."""

    def test__len__(self) -> None:
        """Test __len__."""

        for dist_type, dist_args in zip(distribution_types, distribution_args):
            for size in dataset_sizes:
                for depth in data_depths:
                    for seed in seeds:
                        dataset = SyntheticDataset(
                            size,
                            data_dim,
                            dataset_depth=depth,
                            distribution_type=dist_type,
                            distribution_args=dist_args,
                            seed=seed,
                        )
                        self.assertEqual(len(dataset), size)

    def test__getitem__(self) -> None:
        """Test __getitem__."""

        for dist_type, dist_args in zip(distribution_types, distribution_args):
            for size in dataset_sizes:
                for depth in data_depths:
                    for seed in seeds:
                        dataset = SyntheticDataset(
                            size,
                            data_dim,
                            dataset_depth=depth,
                            distribution_type=dist_type,
                            distribution_args=dist_args,
                            seed=seed,
                        )
                        sample1_1 = dataset[42]
                        sample1_2 = dataset[42]

                        # Test shapes
                        self.assertEqual(sample1_1.shape, sample1_2.shape)
                        self.assertListEqual(list(sample1_1.shape), [depth, data_dim])
                        # Test content
                        if seed < 0:
                            self.assertFalse(torch.equal(sample1_1, sample1_2))
                        else:
                            self.assertTrue(torch.equal(sample1_1, sample1_2))


if __name__ == '__main__':
    unittest.main()
