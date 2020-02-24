"""Testing tmap plotter"""
import unittest
import tempfile
import os
import pandas as pd
import numpy as np
from pytoda.visualizations.molecules.tmap_plotter \
    import tmap, rank_and_normalize_field, thumbnails_with_pubchem_reference


class TestTmapPlotter(unittest.TestCase):

    def _create_test_df(self) -> pd.DataFrame:
        """
        Creates a dataframe to be used on testing

        Returns:
            pd.DataFrame: test dataframe
        """
        return pd.DataFrame(
            [
                ['Cn1cnc2n(C)c(=O)n(C)c(=O)c12', 'green', 0.2],
                [
                    'CCN(CC)C(=O)[C@H]1CN(C)[C@@H]2Cc3c[nH]c4cccc(C2=C1)c34',
                    'red', 0.8
                ],
            ],
            columns=['SMILES', 'cat 1', 'cont 1']
        )

    def test_tmap(self) -> None:
        """Test TMAP wrapper"""
        with tempfile.TemporaryDirectory() as directory:
            tmap(
                self._create_test_df(),
                categorical_columns=['cat 1'],
                continous_columns=['cont 1'],
                plot_folder=directory,
                plot_filename='test'
            )

            self.assertTrue(
                os.path.exists(os.path.join(directory, 'test.html'))
            )

    def test_rank_and_normalize_field(self) -> None:
        data = [-0.5, 0, 1, 1.1]
        expected_output = np.array([0.25, 0.5, 0.75, 1.])

        output = rank_and_normalize_field(data)

        self.assertTrue(np.array_equal(output, expected_output))

    def test_thumbnails_with_pubchem_reference(self) -> None:
        smiles = ['cc1ccc1c', 'c1nC[C]C1n']
        drugs = ['N.A.', 'HEY']
        titles = ['drug 1', 'drug 2']
        expected_labels = [
            'drug 1__No link available__cc1ccc1c',
            (
                'drug 2__<a href="https://pubchem.ncbi.nlm.nih.gov/#query=HEY"'
                '>HEY</a>__c1nC[C]C1n'
            )
        ]

        labels = thumbnails_with_pubchem_reference(smiles, drugs, titles)

        self.assertEqual(labels, expected_labels)


if __name__ == '__main__':
    unittest.main()
