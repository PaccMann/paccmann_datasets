"""Testing Protein Sequence dataset."""
import os
import random
import unittest

from torch.utils.data import DataLoader

from pytoda.datasets import ProteinSequenceDataset
from pytoda.tests.utils import TestFileContent


class TestProteinSequenceDataset(unittest.TestCase):
    """Testing ProteinSequence dataset with eager backend."""

    def test___len__(self) -> None:
        """Test __len__."""
        content = os.linesep.join(
            [
                'EGK	ID3',
                'S	ID1',
                'FGAAV	ID2',
                'NCCS	ID4',
            ]
        )
        with TestFileContent(content) as a_test_file:
            with TestFileContent(content) as another_test_file:
                protein_sequence_dataset = ProteinSequenceDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                )
                self.assertEqual(len(protein_sequence_dataset), 8)

        # Test parsing of .fasta file
        content = r""">sp|Q6GZX0|005R_FRG3G Uncharacterized protein 005R OS=Frog virus 3 (isolate Goorha) OX=654924 GN=FV3-005R PE=4 SV=1
        MQNPLPEVMSPEHDKRTTTPMSKEANKFIRELDKKPGDLAVVSDFVKRNTGKRLPIGKRS
        NLYVRICDLSGTIYMGETFILESWEELYLPEPTKMEVLGTLESCCGIPPFPEWIVMVGED
        QCVYAYGDEEILLFAYSVKQLVEEGIQETGISYKYPDDISDVDEEVLQQDEEIQKIRKKT
        REFVDKDAQEFQDFLNSLDASLLS
        >sp|Q91G88|006L_IIV6 Putative KilA-N domain-containing protein 006L OS=Invertebrate iridescent virus 6 OX=176652 GN=IIV6-006L PE=3 SV=1
        MDSLNEVCYEQIKGTFYKGLFGDFPLIVDKKTGCFNATKLCVLGGKRFVDWNKTLRSKKL
        IQYYETRCDIKTESLLYEIKGDNNDEITKQITGTYLPKEFILDIASWISVEFYDKCNNII
        """

        with TestFileContent(content) as a_test_file:
            protein_sequence_dataset = ProteinSequenceDataset(
                a_test_file.filename, filetype='.fasta'
            )

            self.assertEqual(len(protein_sequence_dataset), 2)

    def test___getitem__(self) -> None:
        """Test __getitem__."""
        content = os.linesep.join(
            [
                'EGK	ID3',
                'S	ID1',
                'FGAAV	ID2',
                'NCCS	ID4',
            ]
        )
        with TestFileContent(content) as a_test_file:
            with TestFileContent(content) as another_test_file:
                protein_sequence_dataset = ProteinSequenceDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    padding=True,
                    add_start_and_stop=True
                )
                pad_index = (
                    protein_sequence_dataset.protein_language.
                    token_to_index['<PAD>']
                )
                start_index = (
                    protein_sequence_dataset.protein_language.
                    token_to_index['<START>']
                )
                stop_index = (
                    protein_sequence_dataset.protein_language.
                    token_to_index['<STOP>']
                )
                e_index = (
                    protein_sequence_dataset.protein_language.
                    token_to_index['E']
                )
                g_index = (
                    protein_sequence_dataset.protein_language.
                    token_to_index['G']
                )
                k_index = (
                    protein_sequence_dataset.protein_language.
                    token_to_index['K']
                )
                n_index = (
                    protein_sequence_dataset.protein_language.
                    token_to_index['N']
                )
                c_index = (
                    protein_sequence_dataset.protein_language.
                    token_to_index['C']
                )
                s_index = (
                    protein_sequence_dataset.protein_language.
                    token_to_index['S']
                )

                self.assertListEqual(
                    protein_sequence_dataset[0].numpy().flatten().tolist(), [
                        pad_index, pad_index, start_index, e_index, g_index,
                        k_index, stop_index
                    ]
                )
                self.assertListEqual(
                    protein_sequence_dataset[7].numpy().flatten().tolist(), [
                        pad_index, start_index, n_index, c_index, c_index,
                        s_index, stop_index
                    ]
                )
                protein_sequence_dataset = ProteinSequenceDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    padding=False,
                    add_start_and_stop=False
                )
                self.assertListEqual(
                    protein_sequence_dataset[0].numpy().flatten().tolist(),
                    [e_index, g_index, k_index]
                )
                self.assertListEqual(
                    protein_sequence_dataset[7].numpy().flatten().tolist(),
                    [n_index, c_index, c_index, s_index]
                )

                # Test padding but no start and stop token
                protein_sequence_dataset = ProteinSequenceDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    padding=True,
                    add_start_and_stop=False,
                )
                self.assertListEqual(
                    protein_sequence_dataset[0].numpy().flatten().tolist(),
                    [pad_index, pad_index, e_index, g_index, k_index]
                )
                self.assertListEqual(
                    protein_sequence_dataset[7].numpy().flatten().tolist(),
                    [pad_index, n_index, c_index, c_index, s_index]
                )

                # Test augmentation / order reversion
                protein_sequence_dataset = ProteinSequenceDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    augment_by_revert=True
                )

                for reverted_sequence in ['S', 'S', 'S', 'S']:
                    token_indexes = (
                        protein_sequence_dataset[1].numpy().flatten().tolist()
                    )
                    sequence = (
                        protein_sequence_dataset.protein_language.
                        token_indexes_to_sequence(token_indexes)
                    )
                    self.assertEqual(sequence, reverted_sequence)
                random.seed(42)
                for reverted_sequence in ['KGE', 'EGK', 'EGK', 'EGK']:
                    token_indexes = (
                        protein_sequence_dataset[0].numpy().flatten().tolist()
                    )
                    sequence = (
                        protein_sequence_dataset.protein_language.
                        token_indexes_to_sequence(token_indexes)
                    )
                    self.assertEqual(sequence, reverted_sequence)

                # Test UNIREP vocab
                protein_sequence_dataset = ProteinSequenceDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    amino_acid_dict='unirep',
                    padding=True,
                    add_start_and_stop=False,
                )
                pad_index = (
                    protein_sequence_dataset.protein_language.
                    token_to_index['<PAD>']
                )
                e_index = (
                    protein_sequence_dataset.protein_language.
                    token_to_index['E']
                )
                g_index = (
                    protein_sequence_dataset.protein_language.
                    token_to_index['G']
                )
                k_index = (
                    protein_sequence_dataset.protein_language.
                    token_to_index['K']
                )
                n_index = (
                    protein_sequence_dataset.protein_language.
                    token_to_index['N']
                )
                c_index = (
                    protein_sequence_dataset.protein_language.
                    token_to_index['C']
                )
                s_index = (
                    protein_sequence_dataset.protein_language.
                    token_to_index['S']
                )
                self.assertListEqual(
                    protein_sequence_dataset[0].numpy().flatten().tolist(),
                    [pad_index, pad_index, e_index, g_index, k_index]
                )
                self.assertListEqual(
                    protein_sequence_dataset[7].numpy().flatten().tolist(),
                    [pad_index, n_index, c_index, c_index, s_index]
                )

        # Test parsing of .fasta file
        content = r""">sp|Q6GZX0|005R_FRG3G Uncharacterized protein 005R OS=Frog virus 3 (isolate Goorha) OX=654924 GN=FV3-005R PE=4 SV=1
        MQNPLPEVMSPEHDKRTTTPMSKEANKFIRELDKKPGDLAVVSDFVKRNTGKRLPIGKRS
        NLYVRICDLSGTIYMGETFILESWEELYLPEPTKMEVLGTLESCCGIPPFPEWIVMVGED
        QCVYAYGDEEILLFAYSVKQLVEEGIQETGISYKYPDDISDVDEEVLQQDEEIQKIRKKT
        REFVDKDAQEFQDFLNSLDASLLS
        >sp|Q91G88|006L_IIV6 Putative KilA-N domain-containing protein 006L OS=Invertebrate iridescent virus 6 OX=176652 GN=IIV6-006L PE=3 SV=1
        MDSLNEVCYEQIKGTFYKGLFGDFPLIVDKKTGCFNATKLCVLGGKRFVDWNKTLRSKKL
        IQYYETRCDIKTESLLYEIKGDNNDEITKQITGTYLPKEFILDIASWISVEFYDKCNNII
        """

        with TestFileContent(content) as a_test_file:
            protein_sequence_dataset = ProteinSequenceDataset(
                a_test_file.filename,
                filetype='.fasta',
                add_start_and_stop=True
            )

            self.assertEqual(len(protein_sequence_dataset[1].tolist()), 206)

    def test_data_loader(self) -> None:
        """Test data_loader."""
        content = os.linesep.join(
            [
                'EGK	ID3',
                'S	ID1',
                'FGAAV	ID2',
                'NCCS	ID4',
            ]
        )
        with TestFileContent(content) as a_test_file:
            with TestFileContent(content) as another_test_file:
                protein_sequence_dataset = ProteinSequenceDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    add_start_and_stop=False
                )
                data_loader = DataLoader(
                    protein_sequence_dataset, batch_size=4, shuffle=True
                )
                for batch_index, batch in enumerate(data_loader):
                    self.assertEqual(batch.shape, (4, 5))
                    if batch_index > 10:
                        break

                protein_sequence_dataset = ProteinSequenceDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    add_start_and_stop=True
                )
                data_loader = DataLoader(
                    protein_sequence_dataset, batch_size=4, shuffle=True
                )
                for batch_index, batch in enumerate(data_loader):
                    self.assertEqual(batch.shape, (4, 7))
                    if batch_index > 10:
                        break


if __name__ == '__main__':
    unittest.main()
