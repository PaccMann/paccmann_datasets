"""Testing SMILESToSMILES dataset."""
import unittest
from functools import wraps

import os

from pytoda.datasets.smiles_to_smiles_dataset import \
    SMILESToSMILESDataset
from pytoda.tests.utils import TestFileContent


def mock_input(fn):

    @wraps(fn)
    def mock_wrapper(self):
        mock_source_file = os.linesep.join(
            [
                'CCO	1',
                'CC	2',
                'CO	3',
                'NCCS	4',
            ]
        )
        mock_target_file = os.linesep.join(
            [
                'CCO[R:1]	1',
                'C([R:1])C	2',
                '[R:1]CO[Q:1]	3',
                'NC([R:1])CS[Q:1]	4',
            ]
        )

        with TestFileContent(mock_source_file) as source_file:
            with TestFileContent(mock_target_file) as target_file:
                fn(self, source_file=source_file, target_file=target_file)

    return mock_wrapper


class TestSMILESToSMILESDataset(unittest.TestCase):
    """Testing the non-annotated polymer dataset"""

    @mock_input
    def test___gettitem__(self, source_file, target_file) -> None:
        ds = SMILESToSMILESDataset(
            input_smi_filepaths=[source_file.filename],
            target_smi_filepaths=[target_file.filename],
            add_start_and_stop=True,
            canonical=True,
            all_bonds_explicit=False,
            sanitize=False,
            padding=False
        )
        smiles_language = ds.smiles_language
        open_p_index = smiles_language.token_to_index['(']
        close_p_index = smiles_language.token_to_index[')']
        c_index = smiles_language.token_to_index['C']
        o_index = smiles_language.token_to_index['O']
        n_index = smiles_language.token_to_index['N']
        s_index = smiles_language.token_to_index['S']
        r1_index = smiles_language.token_to_index['[R:1]']
        q1_index = smiles_language.token_to_index['[Q:1]']

        src, trg = ds[0]

        self.assertEqual(src.tolist(), [2, c_index, c_index, o_index, 3])
        self.assertEqual(
            trg.tolist(), [2, c_index, c_index, o_index, r1_index, 3]
        )

        src, trg = ds[3]
        self.assertEqual(
            src.tolist(), [2, n_index, c_index, c_index, s_index, 3]
        )
        self.assertEqual(
            trg.tolist(), [
                2, n_index, c_index, open_p_index, r1_index, close_p_index,
                c_index, s_index, q1_index, 3
            ]
        )
