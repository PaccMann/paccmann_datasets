"""Testing Protein Sequence dataset."""
import os
import random
import unittest

from torch.utils.data import DataLoader

from pytoda.datasets import ProteinSequenceDataset
from pytoda.tests.utils import TestFileContent

SMI_CONTENT = os.linesep.join(['EGK	ID3', 'S	ID1', 'FGAAV	ID2', 'NCCS	ID4'])
MORE_SMI_CONTENT = os.linesep.join(['KGE	ID5', 'K	ID6', 'SCCN	ID7', 'K	ID8'])


FASTA_CONTENT_UNIPROT = r""">sp|Q6GZX0|005R_FRG3G Uncharacterized protein 005R OS=Frog virus 3 (isolate Goorha) OX=654924 GN=FV3-005R PE=4 SV=1
MQNPLPEVMSPEHDKRTTTPMSKEANKFIRELDKKPGDLAVVSDFVKRNTGKRLPIGKRS
NLYVRICDLSGTIYMGETFILESWEELYLPEPTKMEVLGTLESCCGIPPFPEWIVMVGED
QCVYAYGDEEILLFAYSVKQLVEEGIQETGISYKYPDDISDVDEEVLQQDEEIQKIRKKT
REFVDKDAQEFQDFLNSLDASLLS
>sp|Q91G88|006L_IIV6 Putative KilA-N domain-containing protein 006L OS=Invertebrate iridescent virus 6 OX=176652 GN=IIV6-006L PE=3 SV=1
MDSLNEVCYEQIKGTFYKGLFGDFPLIVDKKTGCFNATKLCVLGGKRFVDWNKTLRSKKL
IQYYETRCDIKTESLLYEIKGDNNDEITKQITGTYLPKEFILDIASWISVEFYDKCNNII
"""  # length 204, 120

FASTA_CONTENT_GENERIC = (
    FASTA_CONTENT_UNIPROT
    + r""">generic_header eager upfp would concat to sequence above.
LLLLLLLLLLLLLLLL
"""
)  # length 16

all_keys = ['ID3', 'ID1', 'ID2', 'ID4', 'Q6GZX0', 'Q91G88']


class TestProteinSequenceDatasetEagerBackend(unittest.TestCase):
    """Testing ProteinSequence dataset with eager backend."""

    def setUp(self):
        self.backend = 'eager'
        print(f'backend is {self.backend}')
        self.smi_content = SMI_CONTENT
        self.smi_other_content = MORE_SMI_CONTENT
        # would fail with FASTA_CONTENT_GENERIC
        self.fasta_content = FASTA_CONTENT_UNIPROT

    def test___len__smi(self) -> None:
        """Test __len__."""

        with TestFileContent(self.smi_content) as a_test_file:
            with TestFileContent(self.smi_other_content) as another_test_file:
                protein_sequence_dataset = ProteinSequenceDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    backend=self.backend,
                )
                self.assertEqual(len(protein_sequence_dataset), 8)

    def test___len__fasta(self) -> None:
        """Test __len__."""
        with TestFileContent(self.fasta_content) as a_test_file:
            protein_sequence_dataset = ProteinSequenceDataset(
                a_test_file.filename, filetype='.fasta', backend=self.backend
            )
            # eager only uniprot headers
            self.assertEqual(len(protein_sequence_dataset), 2)

    def test___getitem__(self) -> None:
        """Test __getitem__."""
        with TestFileContent(self.smi_content) as a_test_file:
            with TestFileContent(self.smi_other_content) as another_test_file:
                protein_sequence_dataset = ProteinSequenceDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    padding=True,
                    add_start_and_stop=True,
                    backend=self.backend,
                )
                pad_index = protein_sequence_dataset.protein_language.token_to_index[
                    '<PAD>'
                ]
                start_index = protein_sequence_dataset.protein_language.token_to_index[
                    '<START>'
                ]
                stop_index = protein_sequence_dataset.protein_language.token_to_index[
                    '<STOP>'
                ]
                e_index = protein_sequence_dataset.protein_language.token_to_index['E']
                g_index = protein_sequence_dataset.protein_language.token_to_index['G']
                k_index = protein_sequence_dataset.protein_language.token_to_index['K']
                n_index = protein_sequence_dataset.protein_language.token_to_index['N']
                c_index = protein_sequence_dataset.protein_language.token_to_index['C']
                s_index = protein_sequence_dataset.protein_language.token_to_index['S']

                self.assertListEqual(
                    protein_sequence_dataset[0].numpy().flatten().tolist(),
                    [
                        pad_index,
                        pad_index,
                        start_index,
                        e_index,
                        g_index,
                        k_index,
                        stop_index,
                    ],
                )
                self.assertListEqual(
                    protein_sequence_dataset[3].numpy().flatten().tolist(),
                    [
                        pad_index,
                        start_index,
                        n_index,
                        c_index,
                        c_index,
                        s_index,
                        stop_index,
                    ],
                )
                self.assertListEqual(
                    protein_sequence_dataset[7].numpy().flatten().tolist(),
                    [
                        pad_index,
                        pad_index,
                        pad_index,
                        pad_index,
                        start_index,
                        k_index,
                        stop_index,
                    ],
                )

                protein_sequence_dataset = ProteinSequenceDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    padding=False,
                    add_start_and_stop=False,
                    backend=self.backend,
                )
                self.assertListEqual(
                    protein_sequence_dataset[0].numpy().flatten().tolist(),
                    [e_index, g_index, k_index],
                )
                self.assertListEqual(
                    protein_sequence_dataset[3].numpy().flatten().tolist(),
                    [n_index, c_index, c_index, s_index],
                )
                self.assertListEqual(
                    protein_sequence_dataset[7].numpy().flatten().tolist(), [k_index]
                )

                # Test padding but no start and stop token
                protein_sequence_dataset = ProteinSequenceDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    padding=True,
                    add_start_and_stop=False,
                    backend=self.backend,
                )
                self.assertListEqual(
                    protein_sequence_dataset[0].numpy().flatten().tolist(),
                    [pad_index, pad_index, e_index, g_index, k_index],
                )
                self.assertListEqual(
                    protein_sequence_dataset[3].numpy().flatten().tolist(),
                    [pad_index, n_index, c_index, c_index, s_index],
                )
                self.assertListEqual(
                    protein_sequence_dataset[7].numpy().flatten().tolist(),
                    [pad_index, pad_index, pad_index, pad_index, k_index],
                )

                # Test augmentation / order reversion
                protein_sequence_dataset = ProteinSequenceDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    augment_by_revert=True,
                )

                random.seed(42)
                for reverted_sequence in ['EGK', 'KGE', 'KGE', 'KGE']:
                    token_indexes = (
                        protein_sequence_dataset[0].numpy().flatten().tolist()
                    )
                    sequence = protein_sequence_dataset.protein_language.token_indexes_to_sequence(
                        token_indexes
                    )
                    self.assertEqual(sequence, reverted_sequence)
                for reverted_sequence in ['S', 'S', 'S', 'S']:
                    token_indexes = (
                        protein_sequence_dataset[1].numpy().flatten().tolist()
                    )
                    sequence = protein_sequence_dataset.protein_language.token_indexes_to_sequence(
                        token_indexes
                    )
                    self.assertEqual(sequence, reverted_sequence)

                # Test UNIREP vocab
                protein_sequence_dataset = ProteinSequenceDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    amino_acid_dict='unirep',
                    padding=True,
                    add_start_and_stop=False,
                    backend=self.backend,
                )
                pad_index = protein_sequence_dataset.protein_language.token_to_index[
                    '<PAD>'
                ]
                e_index = protein_sequence_dataset.protein_language.token_to_index['E']
                g_index = protein_sequence_dataset.protein_language.token_to_index['G']
                k_index = protein_sequence_dataset.protein_language.token_to_index['K']
                n_index = protein_sequence_dataset.protein_language.token_to_index['N']
                c_index = protein_sequence_dataset.protein_language.token_to_index['C']
                s_index = protein_sequence_dataset.protein_language.token_to_index['S']
                self.assertListEqual(
                    protein_sequence_dataset[0].numpy().flatten().tolist(),
                    [pad_index, pad_index, e_index, g_index, k_index],
                )
                self.assertListEqual(
                    protein_sequence_dataset[3].numpy().flatten().tolist(),
                    [pad_index, n_index, c_index, c_index, s_index],
                )
                self.assertListEqual(
                    protein_sequence_dataset[7].numpy().flatten().tolist(),
                    [pad_index, pad_index, pad_index, pad_index, k_index],
                )

        # Test parsing of .fasta file
        with TestFileContent(self.fasta_content) as a_test_file:
            protein_sequence_dataset = ProteinSequenceDataset(
                a_test_file.filename,
                filetype='.fasta',
                add_start_and_stop=True,
                backend=self.backend,
            )
            a_tokenized_sequence = protein_sequence_dataset[1].tolist()
            self.assertEqual(len(a_tokenized_sequence), 206)
            # padded to length + start + stop
            self.assertEqual(sum(a_tokenized_sequence[:-123]), 0)

    def test_data_loader(self) -> None:
        """Test data_loader."""
        with TestFileContent(self.smi_content) as a_test_file:
            with TestFileContent(self.smi_other_content) as another_test_file:
                protein_sequence_dataset = ProteinSequenceDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    add_start_and_stop=False,
                    backend=self.backend,
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
                    add_start_and_stop=True,
                    backend=self.backend,
                )
                data_loader = DataLoader(
                    protein_sequence_dataset, batch_size=4, shuffle=True
                )
                for batch_index, batch in enumerate(data_loader):
                    self.assertEqual(batch.shape, (4, 7))
                    if batch_index > 10:
                        break

    def _test_indexed(self, ds, keys, index):
        key = keys[index]
        positive_index = index % len(ds)
        # get_key (support for negative index?)
        self.assertEqual(key, ds.get_key(positive_index))
        self.assertEqual(key, ds.get_key(index))
        # get_index
        self.assertEqual(positive_index, ds.get_index(key))
        # get_item_from_key
        self.assertTrue(all(ds[index] == ds.get_item_from_key(key)))
        # keys
        self.assertSequenceEqual(keys, list(ds.keys()))
        # duplicate keys
        self.assertFalse(ds.has_duplicate_keys)

    def test_all_base_for_indexed_methods(self):

        with TestFileContent(self.smi_content) as a_test_file:
            with TestFileContent(self.smi_other_content) as another_test_file:
                protein_sequence_ds = ProteinSequenceDataset(
                    a_test_file.filename,
                    another_test_file.filename,
                    backend=self.backend,
                )
                protein_sequence_ds_0 = ProteinSequenceDataset(
                    a_test_file.filename, backend=self.backend
                )
                protein_sequence_ds_1 = ProteinSequenceDataset(
                    another_test_file.filename, backend=self.backend
                )
        all_smiles, all_keys = zip(
            *(
                pair.split('\t')
                for pair in (
                    self.smi_content.split('\n') + self.smi_other_content.split('\n')
                )
            )
        )

        for ds, keys in [
            (protein_sequence_ds, all_keys),
            (protein_sequence_ds_0, all_keys[:4]),
            (protein_sequence_ds_1, all_keys[4:]),
            # no transformation on
            # concat delegation to _SmiLazyDataset/_SmiEagerDataset
            (protein_sequence_ds_0 + protein_sequence_ds_1, all_keys),
        ]:
            index = -1
            self._test_indexed(ds, keys, index)

        # duplicate
        duplicate_ds = protein_sequence_ds_0 + protein_sequence_ds_0
        self.assertTrue(duplicate_ds.has_duplicate_keys)

        # ProteinSequenceDataset tests and raises
        with TestFileContent(self.smi_content) as a_test_file:
            with self.assertRaises(KeyError):
                protein_sequence_ds = ProteinSequenceDataset(
                    a_test_file.filename, a_test_file.filename, backend=self.backend
                )


class TestProteinSequenceDatasetLazyBackend(
    TestProteinSequenceDatasetEagerBackend
):  # noqa
    """Testing ProteinSequence dataset with lazy backend."""

    def setUp(self):
        self.backend = 'lazy'
        print(f'backend is {self.backend}')
        self.smi_content = SMI_CONTENT
        self.smi_other_content = MORE_SMI_CONTENT
        self.fasta_content = FASTA_CONTENT_GENERIC

    def test___len__fasta(self) -> None:
        """Test __len__."""
        with TestFileContent(self.fasta_content) as a_test_file:
            protein_sequence_dataset = ProteinSequenceDataset(
                a_test_file.filename, filetype='.fasta', backend=self.backend
            )
            # generic sequences
            self.assertEqual(len(protein_sequence_dataset), 3)


if __name__ == '__main__':
    unittest.main()
