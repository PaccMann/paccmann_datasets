"""Testing .smi preprocessing utilities."""
import os
import unittest
from io import StringIO

from pytoda.preprocessing.smi import filter_invalid_smi, find_undesired_smiles_files
from pytoda.tests.utils import TestFileContent


class TestSmi(unittest.TestCase):
    """Testing .smi preprocessing."""

    def test_filter_invalid_smi(self) -> None:
        """Test filter_invalid_smi."""
        smiles_content = os.linesep.join(['CCO	compound_a', 'C(	compound_b'])
        filtered_smiles_content = os.linesep.join(['CCO	compound_a'])
        with TestFileContent(smiles_content) as smiles_file:
            with TestFileContent(filtered_smiles_content) as filtered_smiles_file:
                with TestFileContent('') as resulting_smiles_file:
                    print(
                        "\nExpected 'SMILES Parse Error' while filtering "
                        "invalid smiles via rdkit:"
                    )
                    filter_invalid_smi(
                        smiles_file.filename, resulting_smiles_file.filename
                    )
                    with open(resulting_smiles_file.filename) as result_fp:
                        with open(filtered_smiles_file.filename) as filtered_fp:
                            self.assertEqual(
                                result_fp.read().strip(),
                                filtered_fp.read().strip(),
                            )

    def test_find_undesired_smiles_files(self) -> None:
        """Test find_undesired_smiles_files."""

        UNDESIRED = os.linesep.join(['CCO	CHEMBL545', 'NCCS	CHEMBL602'])
        MORE_UNDESIRED = os.linesep.join(['NC(=O)O	CHEMBL123', 'NCCS	CHEMBL602'])
        CONTENT = os.linesep.join(
            [
                'SMILES,ID',
                'COCC(C)N,CHEMBL3184692',
                'COCCOC,CHEMBL1232411',
                'O=CC1CCC1,CHEMBL18475',
                'NC(=O)O,CHEMBL125278',
            ]
        )

        for undesired, gt in zip(
            [UNDESIRED, MORE_UNDESIRED],
            [
                'No matches found, shutting down.\n',
                'Found NC(=O)O in list of undesired SMILES.\n',
            ],
        ):
            with TestFileContent(CONTENT) as content:
                with TestFileContent(undesired) as undesired_content:

                    mystdout = StringIO()
                    find_undesired_smiles_files(
                        undesired_content.filename,
                        content.filename,
                        save_matches=False,
                        file=mystdout,
                    )
                    self.assertEqual(mystdout.getvalue(), gt)


if __name__ == '__main__':
    unittest.main()
