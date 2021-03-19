"""Testing SMILES transforms."""
import unittest

import numpy as np
import torch

from pytoda.smiles.smiles_language import SMILESTokenizer
from pytoda.smiles.transforms import AugmentTensor, Kekulize, NotKekulize, RemoveIsomery


class TestTransforms(unittest.TestCase):
    """Testing transforms."""

    def test_kekulize(self):
        sanitize_opts = [True, False]
        for sanitize in sanitize_opts:
            self._test_kekulize(sanitize)

    def _test_kekulize(self, sanitize) -> None:
        """Test Kekulize."""
        for smiles, ground_truth in [
            ('c1cnoc1', 'C1=CON=C1'),
            ('[O-][n+]1ccccc1S', '[O-][N+]1=CC=CC=C1S'),
            ('c1snnc1-c1ccccn1', 'C1=C(C2=CC=CC=N2)N=NS1'),
        ]:
            transform = Kekulize(
                all_bonds_explicit=False, all_hs_explicit=False, sanitize=sanitize
            )
            self.assertEqual(transform(smiles), ground_truth)

        for smiles, ground_truth in [
            ('c1cnoc1', 'C1=C-O-N=C-1'),
            ('[O-][n+]1ccccc1S', '[O-]-[N+]1=C-C=C-C=C-1-S'),
            ('c1snnc1-c1ccccn1', 'C1=C(-C2=C-C=C-C=N-2)-N=N-S-1'),
        ]:
            transform = Kekulize(
                all_bonds_explicit=True, all_hs_explicit=False, sanitize=sanitize
            )
            self.assertEqual(transform(smiles), ground_truth)

        for smiles, ground_truth in [
            ('c1cnoc1', '[CH]1=[CH][O][N]=[CH]1'),
            ('[O-][n+]1ccccc1S', '[O-][N+]1=[CH][CH]=[CH][CH]=[C]1[SH]'),
            ('c1snnc1-c1ccccn1', '[CH]1=[C]([C]2=[CH][CH]=[CH][CH]=[N]2)[N]=[N][S]1'),
        ]:
            transform = Kekulize(
                all_bonds_explicit=False, all_hs_explicit=True, sanitize=sanitize
            )
            self.assertEqual(transform(smiles), ground_truth)

        for smiles, ground_truth in [
            ('c1cnoc1', '[CH]1=[CH]-[O]-[N]=[CH]-1'),
            ('[O-][n+]1ccccc1S', '[O-]-[N+]1=[CH]-[CH]=[CH]-[CH]=[C]-1-[SH]'),
            (
                'c1snnc1-c1ccccn1',
                '[CH]1=[C](-[C]2=[CH]-[CH]=[CH]-[CH]=[N]-2)-[N]=[N]-[S]-1',
            ),
        ]:
            transform = Kekulize(
                all_bonds_explicit=True, all_hs_explicit=True, sanitize=sanitize
            )
            self.assertEqual(transform(smiles), ground_truth)

    def test_non_kekulize(self) -> None:
        sanitize_opts = [True, False]
        for sanitize in sanitize_opts:
            self._test_non_kekulize(sanitize)

    def _test_non_kekulize(self, sanitize) -> None:
        """Test NotKekulize."""
        for smiles, ground_truth in [
            ('c1cnoc1', 'c1cnoc1'),
            ('[O-][n+]1ccccc1S', '[O-][n+]1ccccc1S'),
            ('c1snnc1-c1ccccn1', 'c1snnc1-c1ccccn1'),
        ]:
            transform = NotKekulize(
                all_bonds_explicit=False, all_hs_explicit=False, sanitize=sanitize
            )
            self.assertEqual(transform(smiles), ground_truth)

        for smiles, ground_truth in [
            ('c1cnoc1', 'c1:c:n:o:c:1'),
            ('[O-][n+]1ccccc1S', '[O-]-[n+]1:c:c:c:c:c:1-S'),
            ('c1snnc1-c1ccccn1', 'c1:s:n:n:c:1-c1:c:c:c:c:n:1'),
        ]:
            transform = NotKekulize(
                all_bonds_explicit=True, all_hs_explicit=False, sanitize=sanitize
            )
            self.assertEqual(transform(smiles), ground_truth)

        for smiles, ground_truth in [
            ('c1cnoc1', '[cH]1[cH][n][o][cH]1'),
            ('[O-][n+]1ccccc1S', '[O-][n+]1[cH][cH][cH][cH][c]1[SH]'),
            ('c1snnc1-c1ccccn1', '[cH]1[s][n][n][c]1-[c]1[cH][cH][cH][cH][n]1'),
        ]:
            transform = NotKekulize(
                all_bonds_explicit=False, all_hs_explicit=True, sanitize=sanitize
            )
            self.assertEqual(transform(smiles), ground_truth)

        for smiles, ground_truth in [
            ('c1cnoc1', '[cH]1:[cH]:[n]:[o]:[cH]:1'),
            ('[O-][n+]1ccccc1S', '[O-]-[n+]1:[cH]:[cH]:[cH]:[cH]:[c]:1-[SH]'),
            (
                'c1snnc1-c1ccccn1',
                '[cH]1:[s]:[n]:[n]:[c]:1-[c]1:[cH]:[cH]:[cH]:[cH]:[n]:1',
            ),
        ]:
            transform = NotKekulize(
                all_bonds_explicit=True, all_hs_explicit=True, sanitize=sanitize
            )
            self.assertEqual(transform(smiles), ground_truth)

    def test_remove_isomery(self) -> None:
        """Test RemoveIsomery."""

        for bonddir, chirality, smiles, ground_truth in zip(
            [False, False, True, True],
            [False, True, False, True],
            4 * ['c1ccc(/C=C/[C@H](C)O)cc1'],
            [
                'c1ccc(/C=C/[C@H](C)O)cc1',
                'c1ccc(/C=C/C(C)O)cc1',
                'c1ccc(C=C[C@H](C)O)cc1',
                'c1ccc(C=CC(C)O)cc1',
            ],
        ):
            transform = RemoveIsomery(
                bonddir=bonddir, chirality=chirality, sanitize=True
            )
            self.assertEqual(transform(smiles), ground_truth)

    def test_augment_tensor(self) -> None:
        sanitize_opts = [True, False]
        for sanitize in sanitize_opts:
            for canonical in [False, True]:
                self._test_augment_tensor(sanitize, canonical)

    def _test_augment_tensor(self, sanitize, canonical) -> None:
        """Test AugmentTensor."""
        smiles = 'NCCS'
        smiles_language = SMILESTokenizer(
            add_start_and_stop=True, padding=False, canonical=canonical
        )
        smiles_language.add_smiles(smiles)

        np.random.seed(0)
        transform = AugmentTensor(smiles_language, sanitize=sanitize)

        def _transform_without_encoding(smile):
            """Encode SMILES by skippping transforms in smiles language. This is
            needed because the new SMILESTokenizer.smiles_to_token_indexes also
            implements SMILES transforms. This could silently canonicalize all SMILES
            if canonical=True in the smiles language object. Hence it's inappropriate
            to generate the ground truth and replace by this method."""
            return smiles_language.transform_encoding(
                [
                    smiles_language.token_to_index[t]
                    for t in smiles_language.smiles_tokenizer(smile)
                ]
            )

        token_indexes_tensor = _transform_without_encoding(smiles)

        for augmented_smile in ['C(S)CN', 'NCCS', 'SCCN', 'C(N)CS', 'C(CS)N']:
            ground_truth = _transform_without_encoding(augmented_smile)
            self.assertSequenceEqual(
                transform(token_indexes_tensor).tolist(), ground_truth.tolist()
            )

        # Now test calling with a tensor of several SMILES
        # Include the padding of the sequence (right padding)
        pl = 5  # padding_length
        single_smiles_tensor = torch.unsqueeze(
            torch.nn.functional.pad(
                token_indexes_tensor, (0, pl), value=smiles_language.padding_index
            ),
            0,
        )
        seq_len = single_smiles_tensor.shape[1]  # sequence_length
        multi_smiles_tensor = torch.cat([single_smiles_tensor] * 5)
        np.random.seed(0)
        augmented = transform(multi_smiles_tensor)

        for ind, augmented_smile in enumerate(
            ['C(S)CN', 'NCCS', 'SCCN', 'C(N)CS', 'C(CS)N']
        ):
            ground_truth = _transform_without_encoding(augmented_smile)
            ground_truth = torch.nn.functional.pad(
                ground_truth,
                pad=(0, seq_len - len(ground_truth)),
                value=smiles_language.padding_index,
            )
            self.assertSequenceEqual(augmented[ind].tolist(), ground_truth.tolist())


if __name__ == '__main__':
    unittest.main()
