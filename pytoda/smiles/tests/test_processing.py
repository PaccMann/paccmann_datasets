"""Testing SMILES processing."""
import unittest
from pytoda.smiles.processing import (
    tokenize_smiles, tokenize_selfies, kmer_smiles_tokenizer
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
                ['[O-]', '[n+]', '1', 'c', 'c', 'c', 'c', 'c', '1', 'S']
            ),
            (
                'c1snnc1-c1ccccn1', [
                    'c', '1', 's', 'n', 'n', 'c', '1', '-', 'c', '1', 'c', 'c',
                    'c', 'c', 'n', '1'
                ]
            )
        ]:
            self.assertListEqual(tokenize_smiles(smiles), ground_truth)

    def test_kmer_smiles_tokenizer(self) -> None:
        """Test kmer_smiles_tokenizer."""

        smiles = [
            'c1ccc(/C=C/[C@H](C)O)cc', '[O-][n+]1ccccc1S', 'c1snnc1-c1ccccn1'
        ]

        k = 1
        stride = 1

        for s in smiles:
            self.assertListEqual(
                tokenize_smiles(s),
                kmer_smiles_tokenizer(s, k=k, stride=stride)
            )

        k = 2
        stride = 2
        for s, gt in zip(
            smiles, [
                ['c1', 'cc', 'c(', '/C', '=C', '/[C@H]', '(C', ')O', ')c'],
                ['[O-][n+]', '1c', 'cc', 'cc', '1S'],
                ['c1', 'sn', 'nc', '1-', 'c1', 'cc', 'cc', 'n1']
            ]
        ):
            self.assertListEqual(
                kmer_smiles_tokenizer(s, k=k, stride=stride), gt
            )

        k = 2
        stride = 1
        for s, gt in zip(
            smiles, [
                [
                    'c1', '1c', 'cc', 'cc', 'c(', '(/', '/C', 'C=', '=C', 'C/',
                    '/[C@H]', '[C@H](', '(C', 'C)', ')O', 'O)', ')c', 'cc'
                ],
                [
                    '[O-][n+]', '[n+]1', '1c', 'cc', 'cc', 'cc', 'cc', 'c1',
                    '1S'
                ],
                [
                    'c1', '1s', 'sn', 'nn', 'nc', 'c1', '1-', '-c', 'c1', '1c',
                    'cc', 'cc', 'cc', 'cn', 'n1'
                ]
            ]
        ):
            self.assertListEqual(
                kmer_smiles_tokenizer(s, k=k, stride=stride), gt
            )

        k = 4
        stride = 4
        for s, gt in zip(
            smiles, [
                ['c1cc', 'c(/C', '=C/[C@H]', '(C)O'], ['[O-][n+]1c', 'cccc'],
                ['c1sn', 'nc1-', 'c1cc', 'ccn1']
            ]
        ):
            self.assertListEqual(
                kmer_smiles_tokenizer(s, k=k, stride=stride), gt
            )

        k = 4
        stride = 2
        for s, gt in zip(
            smiles, [
                [
                    'c1cc', 'ccc(', 'c(/C', '/C=C', '=C/[C@H]', '/[C@H](C',
                    '(C)O', ')O)c'
                ], ['[O-][n+]1c', '1ccc', 'cccc', 'cc1S'],
                ['c1sn', 'snnc', 'nc1-', '1-c1', 'c1cc', 'cccc', 'ccn1']
            ]
        ):
            self.assertListEqual(
                kmer_smiles_tokenizer(s, k=k, stride=stride), gt
            )

    def test_tokenize_selfies(self) -> None:
        """Test tokenize_selfies."""
        for smiles, ground_truth in [
            (
                'c1cnoc1',
                ['[c]', '[c]', '[n]', '[o]', '[c]', '[Ring1]', '[Ring2]']
            ),
            (
                '[O-][n+]1ccccc1S', [
                    '[O-expl]', '[n+expl]', '[c]', '[c]', '[c]', '[c]', '[c]',
                    '[Ring1]', '[Branch1_1]', '[S]'
                ]
            ),
            (
                'c1snnc1-c1ccccn1', [
                    '[c]', '[s]', '[n]', '[n]', '[c]', '[Ring1]', '[Ring2]',
                    '[-c]', '[c]', '[c]', '[c]', '[c]', '[n]', '[Ring1]',
                    '[Branch1_1]'
                ]
            )
        ]:
            transform = Selfies()
            selfies = transform(smiles)
            self.assertListEqual(tokenize_selfies(selfies), ground_truth)


if __name__ == '__main__':
    unittest.main()
