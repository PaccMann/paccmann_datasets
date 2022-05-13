"""Implementation of DrugSensitivityDataset."""
import pandas as pd
import torch
from torch.utils.data import Dataset

from pytoda.warnings import device_warning

from ..smiles.smiles_language import SMILESLanguage
from ..types import DrugSensitivityData, GeneList, Iterable, Tuple
from .gene_expression_dataset import GeneExpressionDataset
from .smiles_dataset import SMILESTokenizerDataset


class DrugSensitivityDataset(Dataset):
    """
    Drug sensitivity dataset implementation.
    """

    def __init__(
        self,
        drug_sensitivity_filepath: str,
        smi_filepath: str,
        gene_expression_filepath: str,
        column_names: Tuple[str] = ['drug', 'cell_line', 'IC50'],
        drug_sensitivity_dtype: torch.dtype = torch.float,
        drug_sensitivity_min_max: bool = True,
        drug_sensitivity_processing_parameters: dict = {},
        smiles_language: SMILESLanguage = None,
        padding: bool = True,
        padding_length: int = None,
        add_start_and_stop: bool = False,
        augment: bool = False,
        canonical: bool = False,
        kekulize: bool = False,
        all_bonds_explicit: bool = False,
        all_hs_explicit: bool = False,
        randomize: bool = False,
        remove_bonddir: bool = False,
        remove_chirality: bool = False,
        selfies: bool = False,
        sanitize: bool = True,
        vocab_file: str = None,
        iterate_dataset: bool = True,
        gene_list: GeneList = None,
        gene_expression_standardize: bool = True,
        gene_expression_min_max: bool = False,
        gene_expression_processing_parameters: dict = {},
        gene_expression_dtype: torch.dtype = torch.float,
        gene_expression_kwargs: dict = {},
        backend: str = 'eager',
        device: torch.device = None,
    ) -> None:
        """
        Initialize a drug sensitivity dataset.

        Args:
            drug_sensitivity_filepath (str): path to drug sensitivity .csv file.
                Currently, the only supported format is .csv, with an index and three
                header columns named as specified in the column_names argument.
            smi_filepath (str): path to .smi file.
            gene_expression_filepath (str): path to gene expression .csv file.
                Currently, the only supported format is .csv,
                with an index and header columns containing gene names.
            column_names (Tuple[str]): Names of columns in data files to retrieve
                labels, ligands and protein name respectively.
                Defaults to ['drug', 'cell_line', 'IC50'].
            drug_sensitivity_dtype (torch.dtype): drug sensitivity data type.
                Defaults to torch.float.
            drug_sensitivity_min_max (bool): min-max scale drug sensitivity
                data. Defaults to True.
            drug_sensitivity_processing_parameters (dict): transformation
                parameters for drug sensitivity data, e.g. for min-max scaling.
                Defaults to {}.
            smiles_language (SMILESLanguage): a smiles language.
                Defaults to None.
            padding (bool): pad sequences to longest in the smiles language.
                Defaults to True.
            padding_length (int): manually sets number of applied paddings,
                applies only if padding is True. Defaults to None.
            add_start_and_stop (bool): add start and stop token indexes.
                Defaults to False.
            canonical (bool): performs canonicalization of SMILES (one
                original string for one molecule), if True, then other
                transformations (augment etc, see below) do not apply
            augment (bool): perform SMILES augmentation. Defaults to False.
            kekulize (bool): kekulizes SMILES (implicit aromaticity only).
                Defaults to False.
            all_bonds_explicit (bool): Makes all bonds explicit. Defaults to
                False, only applies if kekulize = True.
            all_hs_explicit (bool): Makes all hydrogens explicit. Defaults to
                False, only applies if kekulize = True.
            randomize (bool): perform a true randomization of SMILES tokens.
                Defaults to False.
            remove_bonddir (bool): Remove directional info of bonds.
                Defaults to False.
            remove_chirality (bool): Remove chirality information.
                Defaults to False.
            selfies (bool): Whether selfies is used instead of smiles, defaults
                to False.
            sanitize (bool): RDKit sanitization of the molecule.
                Defaults to True.
            vocab_file (str): Optional .json to load vocabulary. Tries to load
                metadata if `iterate_dataset` is False. Defaults to None.
            iterate_dataset (bool): whether to go through all SMILES in the
                dataset to extend/build vocab, find longest sequence, and
                checks the passed padding length if applicable. Defaults to
                True.
            gene_list (GeneList): a list of genes.
            gene_expression_standardize (bool): perform gene expression
                data standardization. Defaults to True.
            gene_expression_min_max (bool): perform min-max scaling on gene
                expression data. Defaults to False.
            gene_expression_processing_parameters (dict): transformation
                parameters for gene expression, e.g. for min-max scaling.
                Defaults to {}.
            gene_expression_dtype (torch.dtype): gene expression data type.
                Defaults to torch.float.
            gene_expression_kwargs (dict): additional parameters for
                GeneExpressionDataset.
            backend (str): memory management backend.
                Defaults to eager, prefer speed over memory consumption.
                Note that at the moment only the gene expression and the
                smiles datasets implement both backends. The drug sensitivity
                data are loaded in memory.
            device (torch.device): DEPRECATED
        """
        Dataset.__init__(self)
        self.drug_sensitivity_filepath = drug_sensitivity_filepath
        self.smi_filepath = smi_filepath
        self.gene_expression_filepath = gene_expression_filepath
        # backend
        self.backend = backend

        if not isinstance(column_names, Iterable):
            raise TypeError(f'Column names was {type(column_names)}, not Iterable.')
        if not len(column_names) == 3:
            raise ValueError(f'Please pass 3 column names not {len(column_names)}')
        self.column_names = column_names
        self.drug_name, self.cell_name, self.label_name = self.column_names
        device_warning(device)

        # SMILES
        self.smiles_dataset = SMILESTokenizerDataset(
            self.smi_filepath,
            smiles_language=smiles_language,
            augment=augment,
            canonical=canonical,
            kekulize=kekulize,
            all_bonds_explicit=all_bonds_explicit,
            all_hs_explicit=all_hs_explicit,
            remove_bonddir=remove_bonddir,
            remove_chirality=remove_chirality,
            selfies=selfies,
            sanitize=sanitize,
            randomize=randomize,
            padding=padding,
            padding_length=padding_length,
            add_start_and_stop=add_start_and_stop,
            vocab_file=vocab_file,
            iterate_dataset=iterate_dataset,
            backend=self.backend,
        )
        # gene expression
        self.gene_expression_dataset = GeneExpressionDataset(
            self.gene_expression_filepath,
            gene_list=gene_list,
            standardize=gene_expression_standardize,
            min_max=gene_expression_min_max,
            processing_parameters=gene_expression_processing_parameters,
            dtype=gene_expression_dtype,
            backend=self.backend,
            index_col=0,
            **gene_expression_kwargs,
        )
        # drug sensitivity
        self.drug_sensitivity_dtype = drug_sensitivity_dtype
        self.drug_sensitivity_min_max = drug_sensitivity_min_max
        self.drug_sensitivity_processing_parameters = (
            drug_sensitivity_processing_parameters
        )
        self.drug_sensitivity_df = pd.read_csv(
            self.drug_sensitivity_filepath, index_col=0
        )
        # filter data based on the availability
        drug_mask = self.drug_sensitivity_df[self.drug_name].isin(
            set(self.smiles_dataset.keys())
        )
        profile_mask = self.drug_sensitivity_df[self.cell_name].isin(
            set(self.gene_expression_dataset.keys())
        )
        self.drug_sensitivity_df = self.drug_sensitivity_df.loc[
            drug_mask & profile_mask
        ]

        # to investigate missing ids per entity
        self.masks_df = pd.concat([drug_mask, profile_mask], axis=1)
        self.masks_df.columns = [self.drug_name, self.cell_name]

        self.number_of_samples = len(self.drug_sensitivity_df)

        # NOTE: optional min-max scaling
        if self.drug_sensitivity_min_max:
            minimum = self.drug_sensitivity_processing_parameters.get(
                'min', self.drug_sensitivity_df[self.label_name].min()
            )
            maximum = self.drug_sensitivity_processing_parameters.get(
                'max', self.drug_sensitivity_df[self.label_name].max()
            )
            self.drug_sensitivity_df[self.label_name] = (
                self.drug_sensitivity_df[self.label_name] - minimum
            ) / (maximum - minimum)
            self.drug_sensitivity_processing_parameters = {
                'processing': 'min_max',
                'parameters': {'min': minimum, 'max': maximum},
            }

    def __len__(self) -> int:
        "Total number of samples."
        return self.number_of_samples

    def __getitem__(self, index: int) -> DrugSensitivityData:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            DrugSensitivityData: a tuple containing three torch.Tensors,
                representing respectively: compound token indexes,
                gene expression values and IC50 for the current sample.
        """
        # drug sensitivity
        selected_sample = self.drug_sensitivity_df.iloc[index]
        ic50_tensor = torch.tensor(
            [selected_sample[self.label_name]],
            dtype=self.drug_sensitivity_dtype,
        )
        # SMILES
        token_indexes_tensor = self.smiles_dataset.get_item_from_key(
            selected_sample[self.drug_name]
        )
        # gene_expression
        gene_expression_tensor = self.gene_expression_dataset.get_item_from_key(
            selected_sample[self.cell_name]
        )
        return token_indexes_tensor, gene_expression_tensor, ic50_tensor
