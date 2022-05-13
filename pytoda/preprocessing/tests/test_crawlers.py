"""Testing Crawlers."""
import unittest

from pytoda.preprocessing.crawlers import (  # query_pubchem,; remove_pubchem_smiles,
    get_smiles_from_pubchem,
    get_smiles_from_zinc,
)


class TestCrawlers(unittest.TestCase):
    """Testing Crawlsers."""

    def test_get_smiles_from_zinc(self) -> None:
        """Test get_smiles_from_zinc"""

        # # ZINC is down since quite some time, hence we skip these tests
        return True

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

        for sanitize in [True, False]:

            # Test text mode
            ground_truth = 'C1=CC(=CC=C1/C=C/C(=O)C2=C(C=C(C=C2)O)O)O'
            for query, drug in zip(['name', 'cid'], ['isoliquiritigenin', 638278]):
                smiles = get_smiles_from_pubchem(
                    drug,
                    use_isomeric=True,
                    kekulize=True,
                    sanitize=sanitize,
                    query_type=query,
                )
            self.assertEqual(smiles, ground_truth)

            ground_truth = 'C1=CC(=CC=C1C=CC(=O)C2=C(C=C(C=C2)O)O)O'
            for query, drug in zip(['name', 'cid'], ['isoliquiritigenin', 638278]):
                smiles = get_smiles_from_pubchem(
                    drug,
                    use_isomeric=False,
                    kekulize=True,
                    sanitize=sanitize,
                    query_type=query,
                )
            # mac-os irreproducible stochastic failure on ubuntu
            self.assertIn(smiles, [ground_truth, ''])

            drug = 'isoliquiritigenin'
            if not sanitize:
                with self.assertRaises(ValueError):
                    get_smiles_from_pubchem(
                        drug, use_isomeric=True, kekulize=False, sanitize=sanitize
                    )

                    get_smiles_from_pubchem(
                        drug, use_isomeric=False, kekulize=False, sanitize=sanitize
                    )
            else:
                ground_truth = 'O=C(/C=C/c1ccc(O)cc1)c1ccc(O)cc1O'

                smiles = get_smiles_from_pubchem(
                    drug, use_isomeric=True, kekulize=False, sanitize=sanitize
                )
                self.assertEqual(smiles, ground_truth)

                ground_truth = 'O=C(C=Cc1ccc(O)cc1)c1ccc(O)cc1O'
                smiles = get_smiles_from_pubchem(
                    drug, use_isomeric=False, kekulize=False, sanitize=sanitize
                )
                self.assertEqual(smiles, ground_truth)

    def test_query_pubchem(self) -> None:
        """Test query_pubchem"""
        pass
        # Disabled due to bug in pubchem api
        # smiles_list = [
        #     'O1C=CC=NC(=O)C1=O',
        #     'CC(N)S(O)(=O)C(C)CC(C(C)C)c1cc(F)cc(F)c1',
        #     'Clc1ccccc2ccnc12',
        # ]
        # ground_truths = [(True, 67945516), (False, -2), (False, -1)]
        # for gt, smiles in zip(ground_truths, smiles_list):
        #     self.assertTupleEqual(query_pubchem(smiles), gt)

    def test_remove_pubchem_smiles(self) -> None:
        """Test remove_pubchem_smiles"""
        pass

        # Disabled due to bug in pubchem api
        # smiles_list = [
        #     'O1C=CC=NC(=O)C1=O',
        #     'CC(N)S(O)(=O)C(C)CC(C(C)C)c1cc(F)cc(F)c1',
        #     'Clc1ccccc2ccnc12',
        # ]
        # ground_truth = ['CC(N)S(O)(=O)C(C)CC(C(C)C)c1cc(F)cc(F)c1', 'Clc1ccccc2ccnc12']
        # self.assertListEqual(remove_pubchem_smiles(smiles_list), ground_truth)


if __name__ == '__main__':
    unittest.main()
