"""Testing DistributionalDataset."""
import unittest

import torch

from pytoda.datasets import DistributionalDataset
from pytoda.datasets.utils.factories import DISTRIBUTION_FUNCTION_FACTORY

distribution_types = ['normal', 'uniform']
distribution_args = [{'loc': 0.0, 'scale': 1.0}, {'low': 0.0, 'high': 1.0}]
dataset_sizes = [100, 10000]
item_shapes = [(1, 16), (10, 16)]
seeds = [None, 1, 42]


class TestDistributionalDataset(unittest.TestCase):
    """Test DistributionalDataset class."""

    def test__len__(self) -> None:
        """Test __len__."""

        for dist_type, dist_args in zip(distribution_types, distribution_args):
            distribution_function = DISTRIBUTION_FUNCTION_FACTORY[dist_type](
                **dist_args
            )
            for size in dataset_sizes:
                for shape in item_shapes:
                    for seed in seeds:
                        dataset = DistributionalDataset(
                            size,
                            shape,
                            distribution_function,
                            seed=seed,
                        )
                        self.assertEqual(len(dataset), size)

    def test__getitem__(self) -> None:
        """Test __getitem__."""

        for dist_type, dist_args in zip(distribution_types, distribution_args):
            distribution_function = DISTRIBUTION_FUNCTION_FACTORY[dist_type](
                **dist_args
            )
            for size in dataset_sizes:
                for shape in item_shapes:
                    for seed in seeds:
                        dataset = DistributionalDataset(
                            size,
                            shape,
                            distribution_function,
                            seed=seed,
                        )
                        sample1_1 = dataset[42]
                        sample1_2 = dataset[42]

                        # Test shapes
                        self.assertEqual(sample1_1.shape, sample1_2.shape)
                        self.assertEqual(sample1_1.shape, shape)
                        # Test content
                        if seed is None:
                            self.assertFalse(torch.equal(sample1_1, sample1_2))
                        else:
                            self.assertTrue(torch.equal(sample1_1, sample1_2))


if __name__ == '__main__':
    unittest.main()
