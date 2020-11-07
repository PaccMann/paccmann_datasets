import torch
import unittest
from torch.utils.data import DataLoader
from pytoda.datasets import SetMatchingDataset, CollatorSetMatching

data_params = {
    "seed": 42,
    "batch_first": "False",
    "dataset_size": 100,
    "max_length": 6,
    "min_length": 2,
    "data_dim": 10,
    "permute": "False",
    "cost_metric": "p-norm",
    "cost_metric_args": {
        "p": 2
    }
}

index1, index2 = torch.randint(0, 100, (2, ))


class TestSetMatchingDataset(unittest.TestCase):

    def setUp(self):
        self.max_length = data_params.get("max_length")
        self.data_dim = data_params.get("data_dim")
        self.batch_first = data_params.get("batch_first")

    def test__len__(self) -> None:
        """Test __len__."""
        setmatch_dataset = SetMatchingDataset(data_params)
        self.assertEqual(len(setmatch_dataset.dataset), 100)

    def test__getitem__(self, index=index1) -> None:
        setmatch_dataset_sampled = SetMatchingDataset(data_params)
        sample1 = setmatch_dataset_sampled.__getitem__(index1)
        self.assertIsInstance(sample1, tuple)

        sample1_set1, sample1_set2, sample1_idx12, sample1_idx21, sample1_length = sample1

        self.assertEqual(
            sample1_length, setmatch_dataset_sampled.set_lengths[index1]
        )
        self.assertEqual(len(sample1_set1), sample1_length)
        self.assertEqual(len(sample1_set1), len(sample1_set2))
        self.assertEqual(len(sample1_idx12), sample1_length)
        self.assertEqual(len(sample1_idx12), len(sample1_idx21))

        same_len_idx = torch.nonzero(
            setmatch_dataset_sampled.set_lengths == sample1_length
        )
        same_len_idx = same_len_idx[same_len_idx != index1]
        index2 = same_len_idx[0]
        sample2 = setmatch_dataset_sampled.__getitem__(index2)
        sample2_set1, sample2_set2, sample2_idx12, sample2_idx21, sample2_length = sample2

        self.assertFalse(torch.equal(sample1_set1, sample2_set1))
        self.assertFalse(torch.equal(sample1_set2, sample2_set2))

        data_params.update({"permute": "True"})

        setmatch_dataset_permuted = SetMatchingDataset(data_params)
        sample3 = setmatch_dataset_permuted.__getitem__(index1)

        sample3_set1, sample3_set2, sample3_idx12, sample3_idx21, sample3_length = sample3

        self.assertTrue(torch.equal(sample3_set1, sample1_set1))

        self.assertTrue(
            torch.equal(sample3_set1, sample3_set2[sample3_idx12, :])
        )
        self.assertTrue(
            torch.equal(sample3_set1[sample3_idx21, :], sample3_set2)
        )

        self.assertTrue(
            torch.equal(
                sample3_idx21,
                setmatch_dataset_permuted.permutation_indices[index1]
            )
        )

    def test_data_loader(self) -> None:
        """Test data_loader."""

        setmatch_dataset_permuted = SetMatchingDataset(data_params)

        collator = CollatorSetMatching(
            self.data_dim, self.max_length, self.batch_first
        )
        data_loader = DataLoader(
            setmatch_dataset_permuted,
            batch_size=25,
            shuffle=True,
            collate_fn=collator
        )
        for batch_index, batch in enumerate(data_loader):
            set1_batch, set2_batch, idx12_batch, idx21_batch, len_batch = batch
            self.assertEqual(
                set1_batch.shape, (25, self.max_length - 1, self.data_dim)
            )
            self.assertEqual(set1_batch.shape, set2_batch.shape)

            self.assertEqual(idx12_batch.shape, idx21_batch.shape)
            self.assertEqual(len(len_batch), 25)

        data_params.update({"permute": "False"})
        setmatch_dataset_sampled = SetMatchingDataset(data_params)

        collator = CollatorSetMatching(
            self.data_dim, self.max_length, self.batch_first
        )
        data_loader = DataLoader(
            setmatch_dataset_sampled,
            batch_size=25,
            shuffle=True,
            collate_fn=collator
        )
        for batch_index, batch in enumerate(data_loader):
            set1_batch, set2_batch, idx12_batch, idx21_batch, len_batch = batch
            self.assertEqual(
                set1_batch.shape, (25, self.max_length - 1, self.data_dim)
            )
            self.assertEqual(set1_batch.shape, set2_batch.shape)

            self.assertEqual(idx12_batch.shape, idx21_batch.shape)
            self.assertEqual(len(len_batch), 25)


if __name__ == '__main__':
    unittest.main()