"""Testing SetMatchingDataset."""
import torch
import unittest
from torch.utils.data import DataLoader
from pytoda.datasets import SetMatchingDataset, CollatorSetMatching, SyntheticDataset

data_params = {
    'seed': [-1, 42],
    'synthetic_seed': [-1, 42],
    'permute': [True, False],
    'vary_set_length': [True, False],
    'dataset_size': 250,
    'data_dim': 4,
    'set_length': 5,
    'batch_first': 'False',
    'cost_metric': 'p-norm',
    'cost_metric_args': {'p': 2},
}

index1 = torch.randint(0, 100, (1,))


class TestSetMatchingDataset(unittest.TestCase):
    """Test SetMatchingDataset class."""

    def test_set_matching_dataset(self) -> None:
        """Test dataset returned by set_matching_dataset."""
        for synthetic_seed in data_params['synthetic_seed']:
            for vary_set_length in data_params['vary_set_length']:
                for seed in data_params['seed']:
                    for permute in data_params['permute']:

                        s1 = SyntheticDataset(
                            data_params['dataset_size'],
                            data_params['data_dim'],
                            dataset_depth=data_params['set_length'],
                            seed=synthetic_seed,
                        )
                        datasets = [s1]
                        if not permute:
                            s2 = SyntheticDataset(
                                data_params['dataset_size'],
                                data_params['data_dim'],
                                dataset_depth=data_params['set_length'],
                                seed=synthetic_seed,
                            )
                            datasets.append(s2)

                        setmatch_dataset = SetMatchingDataset(
                            *datasets,
                            permute=permute,
                            vary_set_length=vary_set_length,
                            seed=seed,
                        )

                        # Test length
                        self.assertEqual(
                            len(setmatch_dataset), data_params['dataset_size']
                        )

                        # Test __getitem__
                        sample1 = setmatch_dataset[0]
                        sample2 = setmatch_dataset[0]

                        self.assertIsInstance(sample1, tuple)
                        self.assertEqual(len(sample1), 4)

                        if synthetic_seed > 0 and seed > 0:
                            for k in range(len(sample1)):
                                self.assertTrue(torch.equal(sample1[k], sample2[k]))
                        elif synthetic_seed > 0:
                            if permute and not vary_set_length:
                                self.assertTrue(
                                    torch.equal(
                                        sample1[1][sample1[2], :],
                                        sample2[1][sample2[2], :],
                                    )
                                )
                                self.assertTrue(
                                    torch.equal(sample1[1][sample1[2], :], sample1[0],)
                                )
                                self.assertTrue(
                                    torch.equal(sample1[0][sample1[3], :], sample1[1],)
                                )
                                self.assertTrue(sample1[0][0, 0] in sample2[1])
                            if not vary_set_length:
                                self.assertTrue(sample1[0][0, 0] in sample2[0][0, 0])
                            else:
                                for k in range(len(sample1)):
                                    self.assertFalse(
                                        torch.equal(sample1[k], sample2[k])
                                    )

                        elif seed > 0:
                            if permute:
                                self.assertTrue(torch.equal(sample1[2], sample2[2]))
                                self.assertTrue(torch.equal(sample1[3], sample2[3]))
                            else:
                                for k in range(len(sample1)):
                                    self.assertFalse(
                                        torch.equal(sample1[k], sample2[k])
                                    )

                        else:
                            for k in range(len(sample1)):

                                self.assertFalse(torch.equal(sample1[k], sample2[k]))

    def test_data_loader(self) -> None:
        """Test data_loader."""

        for synthetic_seed in data_params['synthetic_seed']:
            for vary_set_length in data_params['vary_set_length']:
                for seed in data_params['seed']:
                    for permute in data_params['permute']:

                        s1 = SyntheticDataset(
                            data_params['dataset_size'],
                            data_params['data_dim'],
                            dataset_depth=data_params['set_length'],
                            seed=synthetic_seed,
                        )
                        datasets = [s1]
                        if not permute:
                            s2 = SyntheticDataset(
                                data_params['dataset_size'],
                                data_params['data_dim'],
                                dataset_depth=data_params['set_length'],
                                seed=synthetic_seed + 1,
                            )
                            datasets.append(s2)

                        setmatch_dataset = SetMatchingDataset(
                            *datasets,
                            permute=permute,
                            vary_set_length=vary_set_length,
                            seed=seed,
                        )

                        collator = CollatorSetMatching(
                            data_params['data_dim'],
                            data_params['set_length'],
                            data_params['batch_first'],
                        )
                        data_loader = DataLoader(
                            setmatch_dataset,
                            batch_size=25,
                            shuffle=True,
                            collate_fn=collator,
                        )

                        for batch_index, batch in enumerate(data_loader):
                            (
                                set1_batch,
                                set2_batch,
                                idx12_batch,
                                idx21_batch,
                                len_batch,
                            ) = batch

                            if eval(data_params['batch_first']):
                                self.assertEqual(
                                    set1_batch.shape,
                                    (
                                        25,
                                        data_params['set_length'],
                                        data_params['data_dim'],
                                    ),
                                )
                                self.assertTrue(
                                    torch.unique(set1_batch, dim=0).size(0) == 25
                                )
                                self.assertTrue(
                                    torch.unique(set2_batch, dim=0).size(0) == 25
                                )
                            else:
                                self.assertEqual(
                                    set1_batch.shape,
                                    (
                                        data_params['set_length'],
                                        25,
                                        data_params['data_dim'],
                                    ),
                                )
                                self.assertTrue(
                                    torch.unique(set1_batch, dim=0).size(1) == 25
                                )
                                self.assertTrue(
                                    torch.unique(set2_batch, dim=0).size(1) == 25
                                )
                            self.assertEqual(set1_batch.shape, set2_batch.shape)

                            self.assertEqual(idx12_batch.shape, idx21_batch.shape)
                            self.assertEqual(len(len_batch), 25)

                            self.assertFalse(torch.equal(set1_batch, set2_batch))
                            if permute:

                                if eval(data_params['batch_first']) is False:
                                    set1_batch = set1_batch.permute(1, 0, 2)
                                    set2_batch = set2_batch.permute(1, 0, 2)
                                    self.assertEqual(
                                        set1_batch.size(), set2_batch.size()
                                    )
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

                                self.assertTrue(torch.equal(set1_batch, ordered_set2))
                                self.assertTrue(torch.equal(ordered_set1, set2_batch))


if __name__ == '__main__':
    unittest.main()
