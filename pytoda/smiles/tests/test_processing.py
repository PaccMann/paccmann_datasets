"""Testing SMILES processing."""
import unittest

import selfies as sf

from pytoda.smiles.processing import (
    kmer_smiles_tokenizer,
    spe_smiles_tokenizer,
    split_selfies,
    tokenize_selfies,
    tokenize_smiles,
)
from pytoda.smiles.transforms import Selfies


class TestProcessing(unittest.TestCase):
    """Testing processing."""

    def test_tokenize_smiles(self) -> None:
        """Test tokenize_smiles."""
        for smiles, ground_truth in [
            ('c1cnoc1', ['c', '1', 'c', 'n', 'o', 'c', '1']),
            (
                '[O-][n+]1ccccc1S',
                ['[O-]', '[n+]', '1', 'c', 'c', 'c', 'c', 'c', '1', 'S'],
            ),
            (
                'c1snnc1-c1ccccn1',
                [
                    'c',
                    '1',
                    's',
                    'n',
                    'n',
                    'c',
                    '1',
                    '-',
                    'c',
                    '1',
                    'c',
                    'c',
                    'c',
                    'c',
                    'n',
                    '1',
                ],
            ),
        ]:
            self.assertListEqual(tokenize_smiles(smiles), ground_truth)

    def test_kmer_smiles_tokenizer(self) -> None:
        """Test kmer_smiles_tokenizer."""

        smiles = ['c1ccc(/C=C/[C@H](C)O)cc', '[O-][n+]1ccccc1S', 'c1snnc1-c1ccccn1']

        k = 1
        stride = 1

        for s in smiles:
            self.assertListEqual(
                tokenize_smiles(s), kmer_smiles_tokenizer(s, k=k, stride=stride)
            )

        k = 2
        stride = 2
        for s, gt in zip(
            smiles,
            [
                ['c1', 'cc', 'c(', '/C', '=C', '/[C@H]', '(C', ')O', ')c'],
                ['[O-][n+]', '1c', 'cc', 'cc', '1S'],
                ['c1', 'sn', 'nc', '1-', 'c1', 'cc', 'cc', 'n1'],
            ],
        ):
            self.assertListEqual(kmer_smiles_tokenizer(s, k=k, stride=stride), gt)

        k = 2
        stride = 1
        for s, gt in zip(
            smiles,
            [
                [
                    'c1',
                    '1c',
                    'cc',
                    'cc',
                    'c(',
                    '(/',
                    '/C',
                    'C=',
                    '=C',
                    'C/',
                    '/[C@H]',
                    '[C@H](',
                    '(C',
                    'C)',
                    ')O',
                    'O)',
                    ')c',
                    'cc',
                ],
                ['[O-][n+]', '[n+]1', '1c', 'cc', 'cc', 'cc', 'cc', 'c1', '1S'],
                [
                    'c1',
                    '1s',
                    'sn',
                    'nn',
                    'nc',
                    'c1',
                    '1-',
                    '-c',
                    'c1',
                    '1c',
                    'cc',
                    'cc',
                    'cc',
                    'cn',
                    'n1',
                ],
            ],
        ):
            self.assertListEqual(kmer_smiles_tokenizer(s, k=k, stride=stride), gt)

        k = 4
        stride = 4
        for s, gt in zip(
            smiles,
            [
                ['c1cc', 'c(/C', '=C/[C@H]', '(C)O'],
                ['[O-][n+]1c', 'cccc'],
                ['c1sn', 'nc1-', 'c1cc', 'ccn1'],
            ],
        ):
            self.assertListEqual(kmer_smiles_tokenizer(s, k=k, stride=stride), gt)

        k = 4
        stride = 2
        for s, gt in zip(
            smiles,
            [
                [
                    'c1cc',
                    'ccc(',
                    'c(/C',
                    '/C=C',
                    '=C/[C@H]',
                    '/[C@H](C',
                    '(C)O',
                    ')O)c',
                ],
                ['[O-][n+]1c', '1ccc', 'cccc', 'cc1S'],
                ['c1sn', 'snnc', 'nc1-', '1-c1', 'c1cc', 'cccc', 'ccn1'],
            ],
        ):
            self.assertListEqual(kmer_smiles_tokenizer(s, k=k, stride=stride), gt)

    def test_spe_smiles_tokenizer(self) -> None:
        """Test spe_smiles_tokenizer."""
        for smiles, ground_truth in [
            ('c1ccc(/C=C/[C@H](C)O)cc', ['c1ccc(', '/C=C/', '[C@H](C)', 'O)', 'cc']),
            ('[O-][n+]1ccccc1S', ['[O-]', '[n+]1', 'ccccc1', 'S']),
            ('c1snnc1-c1ccccn1', ['c1s', 'nnc1', '-', 'c1ccccn1']),
        ]:
            self.assertListEqual(spe_smiles_tokenizer(smiles), ground_truth)

    def test_selfies_split(self) -> None:
        """Test tokenization by selfies package has not changed."""
        benzene = 'c1ccccc1'
        encoded_selfies = sf.encoder(benzene)
        # '[c][c][c][c][c][c][Ring1][Branch1_1]' v0.2.4
        # '[C][=C][C][=C][C][=C][Ring1][Branch1_2]' v1.0.2 (no aromatic)

        # sf.split_selfies returns generator
        symbols_benzene = list(sf.split_selfies(encoded_selfies))
        # before selfies 2.0.0 the last token is [Branch1_2]
        self.assertListEqual(
            symbols_benzene,
            ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[=Branch1]'],
        )

        for smiles, ground_truth in [
            (
                'c1cnoc1',
                # before selfies 2.0.0 the 2 last token are [Expl=Ring1] and [Branch1_1]
                ['[C]', '[C]', '[=N]', '[O]', '[C]', '[=Ring1]', '[Branch1]'],
            ),
            (
                '[O-][n+]1ccccc1S',
                # before selfies 2.0.0 it is: [O-expl], [N+expl] and [=Branch1_2]
                [
                    '[O-1]',
                    '[N+1]',
                    '[=C]',
                    '[C]',
                    '[=C]',
                    '[C]',
                    '[=C]',
                    '[Ring1]',
                    '[=Branch1]',
                    '[S]',
                ],
            ),
            (
                'c1snnc1-c1ccccn1',
                # before selfies 2.0.0 it is: [Expl=Ring1], [Branch1_1] and [Branch1_2]
                [
                    '[C]',
                    '[S]',
                    '[N]',
                    '[=N]',
                    '[C]',
                    '[=Ring1]',
                    '[Branch1]',
                    '[C]',
                    '[=C]',
                    '[C]',
                    '[=C]',
                    '[C]',
                    '[=N]',
                    '[Ring1]',
                    '[=Branch1]',
                ],
            ),
        ]:
            self.assertListEqual(
                # list wrapping version
                split_selfies(sf.encoder(smiles)),
                ground_truth,
            )

    def test_tokenize_selfies_match(self) -> None:
        """Test deprecated tokenize_selfies."""
        for smiles in ['c1cnoc1', '[O-][n+]1ccccc1S', 'c1snnc1-c1ccccn1']:
            transform = Selfies()
            selfies = transform(smiles)
            self.assertListEqual(
                tokenize_selfies(selfies), list(sf.split_selfies(selfies))
            )

    def test_package_tokenizer(self) -> None:
        """Test package tokenizer."""
        from pytoda.smiles import vocab

        self.assertEqual(len(vocab), 575)


if __name__ == '__main__':
    unittest.main()
