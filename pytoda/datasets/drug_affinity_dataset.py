"""Implementation of DrugAffinityDataset."""
import torch
import pandas as pd
from torch.utils.data import Dataset
from ..types import DrugAffinityData
from ..smiles.smiles_language import SMILESLanguage
from ..proteins.protein_language import ProteinLanguage
from .smiles_dataset import SMILESDataset
from .protein_sequence_dataset import ProteinSequenceDataset


class DrugAffinityDataset(Dataset):
    """
    Drug affinity dataset implementation.
    """

    def __init__(
        self,
        drug_affinity_filepath: str,
        smi_filepath: str,
        protein_filepath: str,
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
        smiles_selfies: bool = False,
        protein_language: ProteinLanguage = None,
        protein_amino_acid_dict: str = 'iupac',
        protein_padding: bool = True,
        protein_padding_length: int = None,
        protein_add_start_and_stop: bool = False,
        protein_augment_by_revert: bool = False,
        protein_randomize: bool = False,
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
        backend: str = 'eager',
    ) -> None:
        """
        Initialize a drug affinity dataset.

        Args:
            drug_affinity_filepath (str): path to drug affinity
                .csv file. Currently, the only supported format is .csv,
                with an index and three header columns named: "ligand_name",
                "sequence_id", "label".
            smi_filepath (str): path to .smi file.
            protein_filepath (str): path to .smi or .fasta file.
            drug_affinity_dtype (torch.dtype): drug affinity data type.
                Defaults to torch.int.
            smiles_language (SMILESLanguage): a smiles language.
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
            protein_randomize (bool): perform a randomization of the protein
                sequence tokens. Defaults to False.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
            backend (str): memeory management backend.
                Defaults to eager, prefer speed over memory consumption.
                Note that at the moment only the smiles dataset implement both
                backends. The drug affinity data and the protein dataset are
                loaded in memory.
        """
        Dataset.__init__(self)
        self.drug_affinity_filepath = drug_affinity_filepath
        self.smi_filepath = smi_filepath
        self.protein_filepath = protein_filepath
        # device
        self.device = device
        # backend
        self.backend = backend
        # SMILES
        self.smiles_dataset = SMILESDataset(
            self.smi_filepath,
            smiles_language=smiles_language,
            padding=smiles_padding,
            padding_length=smiles_padding_length,
            add_start_and_stop=smiles_add_start_and_stop,
            augment=smiles_augment,
            canonical=smiles_canonical,
            kekulize=smiles_kekulize,
            all_bonds_explicit=smiles_all_bonds_explicit,
            all_hs_explicit=smiles_all_hs_explicit,
            randomize=smiles_randomize,
            remove_bonddir=smiles_remove_bonddir,
            remove_chirality=smiles_remove_chirality,
            selfies=smiles_selfies,
            device=self.device,
            backend=self.backend
        )
        # gene expression
        self.protein_sequence_dataset = ProteinSequenceDataset(
            self.protein_filepath,
            protein_language=protein_language,
            amino_acid_dict=protein_amino_acid_dict,
            padding=protein_padding,
            padding_length=protein_padding_length,
            add_start_and_stop=protein_add_start_and_stop,
            augment_by_revert=protein_augment_by_revert,
            randomize=protein_randomize,
            device=self.device
        )
        # drug affinity
        self.drug_affinity_dtype = drug_affinity_dtype
        self.drug_affinity_df = pd.read_csv(
            self.drug_affinity_filepath, index_col=0
        )
        # NOTE: filter based on the availability
        self.available_drugs = set(
            self.smiles_dataset.sample_to_index_mapping.keys()
        ) & set(self.drug_affinity_df['ligand_name'])
        self.drug_affinity_df = self.drug_affinity_df.loc[
            self.drug_affinity_df['ligand_name'].isin(self.available_drugs)]
        self.available_sequences = set(
            self.protein_sequence_dataset.sample_to_index_mapping.keys()
        ) & set(self.drug_affinity_df['sequence_id'])
        self.drug_affinity_df = self.drug_affinity_df.loc[
            self.drug_affinity_df['sequence_id'].isin(
                self.available_sequences
            )]
        self.number_of_samples = self.drug_affinity_df.shape[0]

    def __len__(self) -> int:
        "Total number of samples."
        return self.number_of_samples

    def __getitem__(self, index: int) -> DrugAffinityData:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            DrugAffinityData: a tuple containing three torch.tensors,
                representing respetively: compound token indexes,
                protein sequence indexes and label for the current sample.
        """
        # drug affinity
        selected_sample = self.drug_affinity_df.iloc[index]
        affinity_tensor = torch.tensor(
            [selected_sample['label']],
            dtype=self.drug_affinity_dtype,
            device=self.device
        )
        # SMILES
        token_indexes_tensor = self.smiles_dataset[
            self.smiles_dataset.sample_to_index_mapping[
                selected_sample['ligand_name']
            ]
        ]  # yapf: disable
        # protein
        protein_sequence_tensor = self.protein_sequence_dataset[
            self.protein_sequence_dataset.sample_to_index_mapping[
                selected_sample['sequence_id']]]
        return token_indexes_tensor, protein_sequence_tensor, affinity_tensor