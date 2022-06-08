"""Implementation of DrugAffinityDataset."""
from typing import Dict, Iterable, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from pytoda.warnings import device_warning

from ..proteins.protein_language import ProteinLanguage
from ..smiles.smiles_language import SMILESLanguage
from ..types import DrugAffinityData
from .protein_sequence_dataset import ProteinSequenceDataset
from .smiles_dataset import SMILESTokenizerDataset


class DrugAffinityDataset(Dataset):
    """
    Drug affinity dataset implementation.
    """

    def __init__(
        self,
        drug_affinity_filepath: str,
        smi_filepath: str,
        protein_filepath: str,
        column_names: Tuple[str] = ['ligand_name', 'sequence_id', 'label'],
        drug_affinity_dtype: torch.dtype = torch.int,
        smiles_language: SMILESLanguage = None,
        smiles_padding: bool = True,
        smiles_padding_length: int = None,
        smiles_add_start_and_stop: bool = False,
        smiles_augment: bool = False,
        smiles_canonical: bool = False,
        smiles_kekulize: bool = False,
        smiles_all_bonds_explicit: bool = False,
        smiles_all_hs_explicit: bool = False,
        smiles_randomize: bool = False,
        smiles_remove_bonddir: bool = False,
        smiles_remove_chirality: bool = False,
        smiles_vocab_file: str = None,
        smiles_selfies: bool = False,
        smiles_sanitize: bool = True,
        protein_language: ProteinLanguage = None,
        protein_amino_acid_dict: str = 'iupac',
        protein_padding: bool = True,
        protein_padding_length: int = None,
        protein_add_start_and_stop: bool = False,
        protein_augment_by_revert: bool = False,
        protein_sequence_augment: Dict = {},
        protein_randomize: bool = False,
        iterate_dataset: bool = True,
        backend: str = 'eager',
        device: torch.device = None,
    ) -> None:
        """
        Initialize a drug affinity dataset.

        Args:
            drug_affinity_filepath (str): path to drug affinity .csv file. Currently,
                the only supported format is .csv, with an index and three header
                columns named as specified in column_names.
            smi_filepath (str): path to .smi file.
            protein_filepath (str): path to .smi or .fasta file.
            column_names (Tuple[str]): Names of columns in data files to retrieve
                labels, ligands and protein name respectively.
                Defaults to ['ligand_name', 'sequence_id', 'label'].
            drug_affinity_dtype (torch.dtype): drug affinity data type.
                Defaults to torch.int.
            smiles_language (SMILESLanguage): a smiles language.
                Defaults to None.
            smiles_vocab_file (str): Optional .json to load vocabulary. Tries
                to load metadata if `iterate_dataset` is False.
                Defaults to None.
            smiles_padding (bool): pad sequences to longest in the smiles
                language. Defaults to True.
            smiles_padding_length (int): manually sets number of applied
                paddings, applies only if padding is True. Defaults to None.
            smiles_add_start_and_stop (bool): add start and stop token indexes.
                Defaults to False.
            smiles_canonical (bool): performs canonicalization of SMILES (one
                original string for one molecule), if True, then other
                transformations (augment etc, see below) do not apply
            smiles_augment (bool): perform SMILES augmentation. Defaults to
                False.
            smiles_kekulize (bool): kekulizes SMILES (implicit aromaticity
                only). Defaults to False.
            smiles_all_bonds_explicit (bool): Makes all bonds explicit.
                Defaults to False, only applies if kekulize = True.
            smiles_all_hs_explicit (bool): Makes all hydrogens explicit.
                Defaults to False, only applies if kekulize = True.
            smiles_randomize (bool): perform a true randomization of SMILES
                tokens. Defaults to False.
            smiles_remove_bonddir (bool): Remove directional info of bonds.
                Defaults to False.
            smiles_remove_chirality (bool): Remove chirality information.
                Defaults to False.
            smiles_selfies (bool): Whether selfies is used instead of smiles.
                Default to False.
            smiles_sanitize (bool): RDKit sanitization of the molecule.
                Defaults to True.
            protein_language (ProteinLanguage): protein language.
                Defaults to None, a.k.a., build it from scratch.
            protein_amino_acid_dict (str): Amino acid dictionary.
                Defaults to 'iupac'.
            protein_padding (bool): pad sequences to the longest in the protein
                language. Defaults to True.
            protein_padding_length (int): manually set the padding.
                Defaults to None.
            protein_add_start_and_stop (bool): add start and stop token
                indexes. Defaults to False.
            protein_augment_by_revert (bool): augment data by reverting the
                sequence. Defaults to False.
            protein_sequence_augment (Dict): a dictionary to specify additional
                sequence augmentation. Defaults to {}.
                NOTE: For details please see `ProteinSequenceDataset`.
            protein_randomize (bool): perform a randomization of the protein
                sequence tokens. Defaults to False.
            protein_vocab_file (str): Optional .json to load vocabulary. Tries
                to load metadata if `iterate_dataset` is False.
                Defaults to None.
            iterate_dataset (bool): whether to go through all items in the
                dataset to extend/build vocab, find longest sequence, and
                checks the passed padding length if applicable. Defaults to
                True.
            backend (str): memory management backend.
                Defaults to eager, prefer speed over memory consumption.
                Note that at the moment only the smiles dataset implement both
                backends. The drug affinity data and the protein dataset are
                loaded in memory.
            device (torch.device): DEPRECATED
        """
        Dataset.__init__(self)
        self.drug_affinity_filepath = drug_affinity_filepath
        self.smi_filepath = smi_filepath
        self.protein_filepath = protein_filepath
        # backend
        self.backend = backend
        device_warning(device)

        if not isinstance(column_names, Iterable):
            raise TypeError(f'Column names was {type(column_names)}, not Iterable.')
        if not len(column_names) == 3:
            raise ValueError(f'Please pass 3 column names not {len(column_names)}')
        self.column_names = column_names
        self.drug_name, self.protein_name, self.label_name = self.column_names

        # SMILES
        self.smiles_dataset = SMILESTokenizerDataset(
            self.smi_filepath,
            smiles_language=smiles_language,
            canonical=smiles_canonical,
            augment=smiles_augment,
            kekulize=smiles_kekulize,
            all_bonds_explicit=smiles_all_bonds_explicit,
            all_hs_explicit=smiles_all_hs_explicit,
            remove_bonddir=smiles_remove_bonddir,
            remove_chirality=smiles_remove_chirality,
            selfies=smiles_selfies,
            sanitize=smiles_sanitize,
            padding=smiles_padding,
            padding_length=smiles_padding_length,
            add_start_and_stop=smiles_add_start_and_stop,
            randomize=smiles_randomize,
            vocab_file=smiles_vocab_file,
            iterate_dataset=iterate_dataset,
            backend=self.backend,
        )
        # protein sequences
        self.protein_sequence_dataset = ProteinSequenceDataset(
            self.protein_filepath,
            protein_language=protein_language,
            amino_acid_dict=protein_amino_acid_dict,
            padding=protein_padding,
            padding_length=protein_padding_length,
            add_start_and_stop=protein_add_start_and_stop,
            augment_by_revert=protein_augment_by_revert,
            sequence_augment=protein_sequence_augment,
            randomize=protein_randomize,
            iterate_dataset=iterate_dataset,
        )
        # drug affinity
        self.drug_affinity_dtype = drug_affinity_dtype
        self.drug_affinity_df = pd.read_csv(self.drug_affinity_filepath, index_col=0)
        # filter data based on the availability
        drug_mask = self.drug_affinity_df[self.drug_name].isin(
            set(self.smiles_dataset.keys())
        )
        sequence_mask = self.drug_affinity_df[self.protein_name].isin(
            set(self.protein_sequence_dataset.keys())
        )
        self.drug_affinity_df = self.drug_affinity_df.loc[drug_mask & sequence_mask]
        # to investigate missing ids per entity
        self.masks_df = pd.concat([drug_mask, sequence_mask], axis=1)
        self.masks_df.columns = [self.drug_name, self.protein_name]

        self.number_of_samples = len(self.drug_affinity_df)

    def __len__(self) -> int:
        "Total number of samples."
        return self.number_of_samples

    def __getitem__(self, index: int) -> DrugAffinityData:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            DrugAffinityData: a tuple containing three torch.Tensors,
                representing respectively: compound token indexes,
                protein sequence indexes and label for the current sample.
        """
        # drug affinity
        selected_sample = self.drug_affinity_df.iloc[index]
        affinity_tensor = torch.tensor(
            [selected_sample[self.label_name]],
            dtype=self.drug_affinity_dtype,
        )
        # SMILES
        token_indexes_tensor = self.smiles_dataset.get_item_from_key(
            selected_sample[self.drug_name]
        )
        # protein
        protein_sequence_tensor = self.protein_sequence_dataset.get_item_from_key(
            selected_sample[self.protein_name]
        )
        return token_indexes_tensor, protein_sequence_tensor, affinity_tensor
