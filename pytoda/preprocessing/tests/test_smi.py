"""Testing .smi preprocessing utilities."""
import unittest
import os
from pytoda.preprocessing.smi import filter_invalid_smi
from pytoda.tests.utils import TestFileContent


class TestSmi(unittest.TestCase):
    """Testing .smi preprocessing."""

    def test_filter_invalid_smi(self) -> None:
        """Test filter_invalid_smi."""
        smiles_content = os.linesep.join(
            [
                'CCO	compound_a',
                'C(	compound_b',
            ]
        )
        filtered_smiles_content = os.linesep.join([
            'CCO	compound_a',
        ])
        with TestFileContent(smiles_content) as smiles_file:
            with TestFileContent(
                filtered_smiles_content
            ) as filtered_smiles_file:
                with TestFileContent('') as resulting_smiles_file:
                    filter_invalid_smi(
                        smiles_file.filename, resulting_smiles_file.filename
                    )
                    with open(resulting_smiles_file.filename) as result_fp:
                        with open(
                            filtered_smiles_file.filename
                        ) as filtered_fp:
                            self.assertEqual(
                                result_fp.read().strip(),
                                filtered_fp.read().strip(),
                            )


if __name__ == '__main__':
    unittest.main()
