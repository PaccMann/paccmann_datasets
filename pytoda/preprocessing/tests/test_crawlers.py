"""Testing Crawlers."""
import unittest
from pytoda.preprocessing.crawlers import (
    get_smiles_from_zinc, get_smiles_from_pubchem
)


class TestCrawlers(unittest.TestCase):
    """Testing Crawlsers."""

    def test_get_smiles_from_zinc(self) -> None:
        """Test get_smiles_from_zinc"""

        # Test text mode
        drug = 'Aspirin'
        ground_truth = 'CC(=O)Oc1ccccc1C(=O)O'
        smiles = get_smiles_from_zinc(drug)
        self.assertEqual(smiles, ground_truth)

        # Test ZINC ID mode
        zinc_id = 53
        ground_truth = 'CC(=O)Oc1ccccc1C(=O)O'
        smiles = get_smiles_from_zinc(zinc_id)
        self.assertEqual(smiles, ground_truth)

    def test_get_smiles_from_pubchem(self) -> None:
        """Test get_smiles_from_zinc"""

        # Test text mode
        drug = 'isoliquiritigenin'
        ground_truth = 'C1=CC(=CC=C1/C=C/C(=O)C2=C(C=C(C=C2)O)O)O'
        smiles = get_smiles_from_pubchem(
            drug, use_isomeric=True, kekulize=True
        )
        self.assertEqual(smiles, ground_truth)

        ground_truth = 'C1=CC(=CC=C1C=CC(=O)C2=C(C=C(C=C2)O)O)O'
        smiles = get_smiles_from_pubchem(
            drug, use_isomeric=False, kekulize=True
        )
        self.assertEqual(smiles, ground_truth)

        ground_truth = 'O=C(/C=C/c1ccc(O)cc1)c1ccc(O)cc1O'
        smiles = get_smiles_from_pubchem(
            drug, use_isomeric=True, kekulize=False
        )
        self.assertEqual(smiles, ground_truth)

        ground_truth = 'O=C(C=Cc1ccc(O)cc1)c1ccc(O)cc1O'
        smiles = get_smiles_from_pubchem(
            drug, use_isomeric=False, kekulize=False
        )
        self.assertEqual(smiles, ground_truth)


if __name__ == '__main__':
    unittest.main()
