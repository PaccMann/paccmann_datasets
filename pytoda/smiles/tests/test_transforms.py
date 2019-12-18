"""Testing SMILES transforms."""
import unittest
from pytoda.smiles.transforms import RemoveIsomery, Kekulize
from pytoda.smiles.transforms import Selfies


class TestTransforms(unittest.TestCase):
    """Testing transforms."""

    def test_kekulize(self) -> None:
        """Test Kekulize."""
        for smiles, ground_truth in [
            ('c1cnoc1', 'C1=CON=C1'),
            ('[O-][n+]1ccccc1S', '[O-][N+]1=CC=CC=C1S'),
            ('c1snnc1-c1ccccn1', 'C1=C(C2=CC=CC=N2)N=NS1')
        ]:
            transform = Kekulize(
                all_bonds_explicit=False, all_hs_explicit=False
            )
            self.assertEqual(transform(smiles), ground_truth)

        for smiles, ground_truth in [
            ('c1cnoc1', 'C1=C-O-N=C-1'),
            ('[O-][n+]1ccccc1S', '[O-]-[N+]1=C-C=C-C=C-1-S'),
            ('c1snnc1-c1ccccn1', 'C1=C(-C2=C-C=C-C=N-2)-N=N-S-1')
        ]:
            transform = Kekulize(
                all_bonds_explicit=True, all_hs_explicit=False
            )
            self.assertEqual(transform(smiles), ground_truth)

        for smiles, ground_truth in [
            ('c1cnoc1', '[CH]1=[CH][O][N]=[CH]1'),
            ('[O-][n+]1ccccc1S', '[O-][N+]1=[CH][CH]=[CH][CH]=[C]1[SH]'),
            (
                'c1snnc1-c1ccccn1',
                '[CH]1=[C]([C]2=[CH][CH]=[CH][CH]=[N]2)[N]=[N][S]1'
            )
        ]:
            transform = Kekulize(
                all_bonds_explicit=False, all_hs_explicit=True
            )
            self.assertEqual(transform(smiles), ground_truth)

        for smiles, ground_truth in [
            ('c1cnoc1', '[CH]1=[CH]-[O]-[N]=[CH]-1'),
            ('[O-][n+]1ccccc1S', '[O-]-[N+]1=[CH]-[CH]=[CH]-[CH]=[C]-1-[SH]'),
            (
                'c1snnc1-c1ccccn1',
                '[CH]1=[C](-[C]2=[CH]-[CH]=[CH]-[CH]=[N]-2)-[N]=[N]-[S]-1'
            )
        ]:
            transform = Kekulize(all_bonds_explicit=True, all_hs_explicit=True)
            self.assertEqual(transform(smiles), ground_truth)

    def test_remove_isomery(self) -> None:
        """Test RemoveIsomery."""

        for bonddir, chirality, smiles, ground_truth in zip(
            [False, False, True, True],
            [False, True, False, True],
            4 * ['C/C=C/C[C@H](O)Cc1ccccc1'],
            [
                'C/C=C/C[C@H](O)Cc1ccccc1', 'C/C=C/CC(O)Cc1ccccc1',
                'CC=CC[C@H](O)Cc1ccccc1', 'CC=CCC(O)Cc1ccccc1'
            ]
        ):  # yapf: disable
            transform = RemoveIsomery(bonddir=bonddir, chirality=chirality)
            self.assertEqual(transform(smiles), ground_truth)


if __name__ == '__main__':
    unittest.main()
