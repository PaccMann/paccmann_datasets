"""Testing files processing."""
import os
import unittest

from pytoda.files import count_file_lines, read_smi
from pytoda.tests.utils import TestFileContent


class TestFiles(unittest.TestCase):
    """Testing files."""

    def test_count_file_lines(self) -> None:
        """Test count_file_lines."""
        content = os.linesep.join(['one batch', 'two batch', 'penny and dime'])
        with TestFileContent(content) as test_file:
            self.assertEqual(count_file_lines(test_file.filename), 3)
        content = os.linesep.join(
            ['one batch', 'two batch', 'penny and dime{}'.format(os.linesep)]
        )
        with TestFileContent(content) as test_file:
            self.assertEqual(count_file_lines(test_file.filename), 3)

    def test_read_smi(self) -> None:
        """Test read_smi."""
        content = os.linesep.join(
            [
                'CN(C)CCNC(=O)c1cc2CSc3cc(Cl)ccc3-c2s1\tdrug_0',
                'CC(C)N1C(=O)S\\C(=C\\c2ccc(Sc3nc4ccccc4[nH]3)o2)C1=O\tdrug_1',
                'C(Cn1c2ccccc2c2ccccc12)c1nc2ccccc2[nH]1\tdrug_2',
                'C1CN(CCO1)c1nnc(-c2ccccc2)c(n1)-c1ccccc1\tdrug_3',
            ]
        )
        indexes = ['drug_0', 'drug_1', 'drug_2', 'drug_3']
        smiles = [
            'CN(C)CCNC(=O)c1cc2CSc3cc(Cl)ccc3-c2s1',
            'CC(C)N1C(=O)S\\C(=C\\c2ccc(Sc3nc4ccccc4[nH]3)o2)C1=O',
            'C(Cn1c2ccccc2c2ccccc12)c1nc2ccccc2[nH]1',
            'C1CN(CCO1)c1nnc(-c2ccccc2)c(n1)-c1ccccc1',
        ]
        with TestFileContent(content) as test_file:
            smiles_df = read_smi(test_file.filename)
            self.assertEqual(smiles_df.shape, (4, 1))
            self.assertEqual(smiles_df.index.tolist(), indexes)
            for index, a_smiles in zip(indexes, smiles):
                self.assertEqual(smiles_df.loc[index]['SMILES'], a_smiles)


if __name__ == '__main__':
    unittest.main()
