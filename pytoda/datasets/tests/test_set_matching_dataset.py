"""Testing SetMatchingDataset."""
import unittest
from typing import List

import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from pytoda.datasets import (
    DistributionalDataset,
    PairedSetMatchingDataset,
    PermutedSetMatchingDataset,
)
from pytoda.datasets.utils.factories import (
    DISTRIBUTION_FUNCTION_FACTORY,
    METRIC_FUNCTION_FACTORY,
)

seeds = [None, 42]
distribution_seeds = [None, 42]

min_set_length = [5, 2]
set_padding_value = 6.0
dataset_size = 250
max_set_length = 10
item_shape = (max_set_length, 4)
cost_metric = 'p-norm'
cost_metric_args = {'p': 2}
distribution_type = ['normal', 'uniform']
distribution_args = [{'loc': 0.0, 'scale': 1.0}, {'low': 0, 'high': 1}]
noise_std = [0.001, 0.1]  # Pytorch >1.7 errors with a noise of 0.0.

cost_metric_function = METRIC_FUNCTION_FACTORY[cost_metric](**cost_metric_args)

permute = [True, False]
DATASET_FACTORY = {True: PermutedSetMatchingDataset, False: PairedSetMatchingDataset}


class TestSetMatchingDataset(unittest.TestCase):
    """Test SetMatchingDataset class."""

    def test_permuted_set_matching_dataset(self) -> None:
        """Test PermutedSetMatchingDataset class."""

        def tolist(x: torch.Tensor) -> List:
            return x.flatten().tolist()

        for dist_type, dist_args in zip(distribution_type, distribution_args):

            distribution_function = DISTRIBUTION_FUNCTION_FACTORY[dist_type](
                **dist_args
            )
            for dist_seed in distribution_seeds:
                for seed in seeds:
                    for noise in noise_std:
                        for min_len in min_set_length:

                            s1 = DistributionalDataset(
                                dataset_size,
                                item_shape,
                                distribution_function,
                                seed=dist_seed,
                            )
                            datasets = [s1]

                            permuted_dataset = PermutedSetMatchingDataset(
                                *datasets,
                                min_len,
                                cost_metric_function,
                                set_padding_value=set_padding_value,
                                noise_std=noise,
                                seed=seed,
                            )
                            # Test length
                            self.assertEqual(len(permuted_dataset), dataset_size)

                            # Test __getitem__
                            sample1 = permuted_dataset[0]
                            sample2 = permuted_dataset[0]

                            self.assertIsInstance(sample1, tuple)
                            self.assertEqual(len(sample1), 5)

                            if dist_seed is not None and seed is not None:
                                # since both distribution seed and permutation seed
                                # are fixed, sampling twice with same index should return
                                # identical samples.

                                for item1, item2 in zip(sample1, sample2):
                                    self.assertTrue(torch.equal(item1, item2))

                                self.assertTrue(
                                    torch.equal(
                                        sample1[1][sample1[2].long(), :],
                                        sample2[1][sample2[2].long(), :],
                                    )
                                )

                                # When noise =0, permutation of set2 should return set1
                                # and vice versa
                                if noise < 0.01:

                                    for a, b in zip(
                                        tolist(sample1[1][sample1[2].long(), :]),
                                        tolist(sample1[0]),
                                    ):
                                        self.assertAlmostEqual(a, b, places=2)
                                    for a, b in zip(
                                        tolist(sample1[0][sample1[3].long(), :]),
                                        tolist(sample1[1]),
                                    ):
                                        self.assertAlmostEqual(a, b, places=2)

                                else:

                                    self.assertFalse(
                                        torch.equal(
                                            sample1[1][sample1[2].long(), :],
                                            sample1[0],
                                        )
                                    )
                                    self.assertFalse(
                                        torch.equal(
                                            sample1[0][sample1[3].long(), :],
                                            sample1[1],
                                        )
                                    )

                            elif dist_seed is not None:
                                # Since only distribution seed is fixed, for fixed length
                                # settings the reference set returned at index 0 must be identical.
                                # NOTE: since items are padded when lengths vary,
                                # a lower limit on max set length is required to test
                                # that permutations are not equal when permutation seed is None.

                                if min_len == max_set_length:
                                    self.assertTrue(torch.equal(sample1[0], sample2[0]))
                                    if max_set_length > 3:
                                        self.assertFalse(
                                            torch.equal(sample1[2], sample2[2]),
                                            msg=f'{sample1},{sample2}',
                                        )

                                    if noise < 0.01:
                                        for a, b in zip(
                                            tolist(sample1[1][sample1[2].long(), :]),
                                            tolist(sample2[1][sample2[2].long(), :]),
                                        ):
                                            self.assertAlmostEqual(a, b, places=2)

                                elif sample1[-1] != sample2[-1]:
                                    # reason for asserting false is that length
                                    # cropping is a random event dependent on
                                    # permutation seed which is None in this setting

                                    self.assertFalse(
                                        torch.equal(sample1[0], sample2[0]),
                                        msg=f'{sample1},{sample2}',
                                    )

                                self.assertFalse(
                                    torch.equal(sample1[1], sample2[1]),
                                )

                                if noise < 0.01:
                                    for a, b in zip(
                                        tolist(sample1[1][sample1[2].long(), :]),
                                        tolist(sample1[0]),
                                    ):
                                        self.assertAlmostEqual(a, b, places=2)

                                    for a, b in zip(
                                        tolist(sample1[0][sample1[3].long(), :]),
                                        tolist(sample1[1]),
                                    ):
                                        self.assertAlmostEqual(a, b, places=2)

                                else:
                                    self.assertFalse(
                                        torch.equal(
                                            sample1[1][sample1[2].long(), :], sample1[0]
                                        )
                                    )
                                    self.assertFalse(
                                        torch.equal(
                                            sample1[0][sample1[3].long(), :], sample1[1]
                                        )
                                    )

                            elif seed is not None:
                                # since distribution seed is None, the sampled sets
                                # must be different but the lengths and permutations
                                # should be the same since the permutation seed is set.

                                self.assertFalse(torch.equal(sample1[0], sample2[0]))
                                self.assertFalse(torch.equal(sample1[1], sample2[1]))

                                self.assertTrue(torch.equal(sample1[-1], sample2[-1]))
                                if noise < 0.01:

                                    for a, b in zip(
                                        tolist(sample1[2]),
                                        tolist(sample2[2]),
                                    ):
                                        self.assertAlmostEqual(a, b, places=2)

                                    for a, b in zip(
                                        tolist(sample1[1][sample1[2].long(), :]),
                                        tolist(sample1[0]),
                                    ):
                                        self.assertAlmostEqual(a, b, places=2)

                                    for a, b in zip(
                                        tolist(sample1[0][sample1[3].long(), :]),
                                        tolist(sample1[1]),
                                    ):
                                        self.assertAlmostEqual(a, b, places=2)

                                else:
                                    self.assertFalse(
                                        torch.equal(
                                            sample1[1][sample1[2].long(), :],
                                            sample1[0],
                                        )
                                    )
                                    self.assertFalse(
                                        torch.equal(
                                            sample1[0][sample1[3].long(), :],
                                            sample1[1],
                                        )
                                    )

                            else:
                                # since both seeds are none, the sets and cropped lengths
                                # must be different. Difference in permutations are only
                                # checked if length>3 due to item-wise padding.
                                self.assertFalse(torch.equal(sample1[0], sample2[0]))
                                self.assertFalse(torch.equal(sample1[1], sample2[1]))

    def test_paired_set_matching_dataset(self) -> None:
        """Test PairedSetMatchingDataset class."""
        # Similar reasoning with respect to seeds follows from above.
        # Main difference is that the hungarian assignments, i.e, the targets
        # returned are not tested since these assignments are not dependent on
        # the permutation seed but only on the pair of sets generated.

        noise = 0.0

        for dist_type, dist_args in zip(distribution_type, distribution_args):

            distribution_function = DISTRIBUTION_FUNCTION_FACTORY[dist_type](
                **dist_args
            )
            for dist_seed in distribution_seeds:
                for seed in seeds:
                    for min_len in min_set_length:

                        if dist_seed is not None:
                            seed_s1 = dist_seed
                            seed_s2 = dist_seed + 1
                        else:
                            seed_s1 = seed_s2 = dist_seed

                        s1 = DistributionalDataset(
                            dataset_size,
                            item_shape,
                            distribution_function,
                            seed=seed_s1,
                        )
                        datasets = [s1]

                        s2 = DistributionalDataset(
                            dataset_size,
                            item_shape,
                            distribution_function,
                            seed=seed_s2,
                        )
                        datasets.append(s2)

                        paired_dataset = PairedSetMatchingDataset(
                            *datasets,
                            min_len,
                            cost_metric_function,
                            set_padding_value=set_padding_value,
                            noise_std=noise,
                            seed=seed,
                        )
                        # Test length
                        self.assertEqual(len(paired_dataset), dataset_size)

                        # Test __getitem__
                        sample1 = paired_dataset[0]
                        sample2 = paired_dataset[0]

                        sample1_hungarian12 = linear_sum_assignment(
                            torch.cdist(sample1[0], sample1[1]).numpy()
                        )[1]

                        sample1_hungarian21 = linear_sum_assignment(
                            torch.cdist(sample1[1], sample1[0]).numpy()
                        )[1]

                        self.assertTrue(
                            torch.equal(
                                sample1[2].int(),
                                torch.from_numpy(sample1_hungarian12).int(),
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                sample1[3].int(),
                                torch.from_numpy(sample1_hungarian21).int(),
                            )
                        )

                        self.assertIsInstance(sample1, tuple)
                        self.assertEqual(len(sample1), 5)
                        self.assertFalse(torch.equal(sample1[0], sample1[1]))

                        if dist_seed is not None and seed is not None:

                            for item1, item2 in zip(sample1, sample2):
                                self.assertTrue(torch.equal(item1, item2))

                        elif dist_seed is not None:

                            if min_len == max_set_length:
                                for item1, item2 in zip(sample1, sample2):
                                    self.assertTrue(torch.equal(item1, item2))

                            elif sample1[-1] != sample2[-1]:
                                for item1, item2 in zip(sample1[:2], sample2[:2]):
                                    self.assertFalse(torch.equal(item1, item2))

                        elif seed is not None:

                            self.assertFalse(torch.equal(sample1[0], sample2[0]))
                            self.assertFalse(torch.equal(sample1[1], sample2[1]))

                            self.assertTrue(torch.equal(sample1[-1], sample2[-1]))

                        else:
                            for item1, item2 in zip(sample1[:2], sample2[:2]):
                                self.assertFalse(torch.equal(item1, item2))

    def test_data_loader(self) -> None:
        """Test data_loader or SetMatchingDataset."""

        for dist_seed in distribution_seeds:
            for dist_type, dist_args in zip(distribution_type, distribution_args):
                distribution_function = DISTRIBUTION_FUNCTION_FACTORY[dist_type](
                    **dist_args
                )
                for noise in noise_std:
                    for seed in seeds:
                        for permute_ in permute:
                            for min_len in min_set_length:
                                if dist_seed is None:
                                    seed_s1 = seed_s2 = dist_seed

                                else:
                                    seed_s1 = dist_seed
                                    seed_s2 = dist_seed + 1

                                s1 = DistributionalDataset(
                                    dataset_size,
                                    item_shape,
                                    distribution_function,
                                    seed=seed_s1,
                                )
                                datasets = [s1]
                                if not permute_:
                                    s2 = DistributionalDataset(
                                        dataset_size,
                                        item_shape,
                                        distribution_function,
                                        seed=seed_s2,
                                    )
                                    datasets.append(s2)

                                setmatch_dataset = DATASET_FACTORY[permute_](
                                    *datasets,
                                    min_len,
                                    cost_metric_function,
                                    set_padding_value=set_padding_value,
                                    noise_std=noise,
                                    seed=seed,
                                )

                                data_loader = DataLoader(
                                    setmatch_dataset,
                                    batch_size=25,
                                )

                                for batch_index, batch in enumerate(data_loader):
                                    (
                                        set1_batch,
                                        set2_batch,
                                        idx12_batch,
                                        idx21_batch,
                                        len_batch,
                                    ) = batch

                                    self.assertEqual(
                                        set1_batch.shape,
                                        (
                                            25,
                                            max_set_length,
                                            4,
                                        ),
                                    )
                                    self.assertTrue(
                                        torch.unique(set1_batch, dim=0).size(0) == 25
                                    )
                                    self.assertTrue(
                                        torch.unique(set2_batch, dim=0).size(0) == 25
                                    )

                                    self.assertEqual(set1_batch.shape, set2_batch.shape)

                                    self.assertEqual(
                                        idx12_batch.shape, idx21_batch.shape
                                    )
                                    self.assertEqual(len(len_batch), 25)

                                    self.assertFalse(
                                        torch.equal(set1_batch, set2_batch)
                                    )

                                    if permute_ and noise_std == 0.0:

                                        ordered_set1 = set1_batch[
                                            torch.arange(0, idx21_batch.size(0))
                                            .unsqueeze(1)
                                            .repeat((1, idx21_batch.size(1))),
                                            idx21_batch,
                                        ]

                                        ordered_set2 = set2_batch[
                                            torch.arange(0, idx12_batch.size(0))
                                            .unsqueeze(1)
                                            .repeat((1, idx12_batch.size(1))),
                                            idx12_batch,
                                        ]

                                        self.assertTrue(
                                            torch.equal(set1_batch, ordered_set2)
                                        )
                                        self.assertTrue(
                                            torch.equal(ordered_set1, set2_batch)
                                        )


if __name__ == '__main__':
    unittest.main()
