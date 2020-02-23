"""Implementation of DrugSensitivityDataset."""
import torch
import pandas as pd
from torch.utils.data import Dataset
from ..types import GeneList, DrugSensitivityData
from ..smiles.smiles_language import SMILESLanguage
from .smiles_dataset import SMILESDataset
from .gene_expression_dataset import GeneExpressionDataset


class DrugSensitivityDataset(Dataset):
    """
    Drug sensitivity dataset implementation.
    """

    def __init__(
        self,
        drug_sensitivity_filepath: str,
        smi_filepath: str,
        gene_expression_filepath: str,
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
        gene_list: GeneList = None,
        gene_expression_standardize: bool = True,
        gene_expression_min_max: bool = False,
        gene_expression_processing_parameters: dict = {},
        gene_expression_dtype: torch.dtype = torch.float,
        gene_expression_kwargs: dict = {},
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
        backend: str = 'eager',
    ) -> None:
        """
        Initialize a drug sensitivity dataset.

        Args:
            drug_sensitivity_filepath (str): path to drug sensitivity
                .csv file. Currently, the only supported format is .csv,
                with an index and three header columns named: "drug",
                "cell_line", "IC50".
            smi_filepath (str): path to .smi file.
            gene_expression_filepath (str): path to gene expression .csv file.
                Currently, the only supported format is .csv,
                with an index and header columns containing gene names.
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
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
            backend (str): memeory management backend.
                Defaults to eager, prefer speed over memory consumption.
                Note that at the moment only the gene expression and the
                smiles datasets implement both backends. The drug sensitivity
                data are loaded in memory.
        """
        Dataset.__init__(self)
        self.drug_sensitivity_filepath = drug_sensitivity_filepath
        self.smi_filepath = smi_filepath
        self.gene_expression_filepath = gene_expression_filepath
        # device
        self.device = device
        # backend
        self.backend = backend
        # SMILES
        self.smiles_dataset = SMILESDataset(
            self.smi_filepath,
            smiles_language=smiles_language,
            padding=padding,
            padding_length=padding_length,
            add_start_and_stop=add_start_and_stop,
            augment=augment,
            canonical=canonical,
            kekulize=kekulize,
            all_bonds_explicit=all_bonds_explicit,
            all_hs_explicit=all_hs_explicit,
            randomize=randomize,
            remove_bonddir=remove_bonddir,
            remove_chirality=remove_chirality,
            selfies=selfies,
            device=self.device,
            backend=self.backend
        )
        # gene expression
        self.gene_expression_dataset = GeneExpressionDataset(
            self.gene_expression_filepath,
            gene_list=gene_list,
            standardize=gene_expression_standardize,
            min_max=gene_expression_min_max,
            processing_parameters=gene_expression_processing_parameters,
            dtype=gene_expression_dtype,
            device=self.device,
            backend=self.backend,
            index_col=0,
            **gene_expression_kwargs
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
        # NOTE: filter based on the availability
        self.available_drugs = set(
            self.smiles_dataset.sample_to_index_mapping.keys()
        ) & set(self.drug_sensitivity_df['drug'])
        self.drug_sensitivity_df = self.drug_sensitivity_df.loc[
            self.drug_sensitivity_df['drug'].isin(self.available_drugs)]
        self.available_profiles = set(
            self.gene_expression_dataset.sample_to_index_mapping.keys()
        ) & set(self.drug_sensitivity_df['cell_line'])
        self.drug_sensitivity_df = self.drug_sensitivity_df.loc[
            self.drug_sensitivity_df['cell_line'].isin(
                self.available_profiles
            )]
        self.number_of_samples = self.drug_sensitivity_df.shape[0]
        # NOTE: optional min-max scaling
        if self.drug_sensitivity_min_max:
            minimum = (
                self.drug_sensitivity_processing_parameters.get(
                    'min', self.drug_sensitivity_df['IC50'].min()
                )
            )
            maximum = (
                self.drug_sensitivity_processing_parameters.get(
                    'max', self.drug_sensitivity_df['IC50'].max()
                )
            )
            self.drug_sensitivity_df['IC50'] = (
                (self.drug_sensitivity_df['IC50'] - minimum) /
                (maximum - minimum)
            )
            self.drug_sensitivity_processing_parameters = {
                'processing': 'min_max',
                'parameters': {
                    'min': minimum,
                    'max': maximum
                }
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
            DrugSensitivityData: a tuple containing three torch.tensors,
                representing respetively: compound token indexes,
                gene expression values and IC50 for the current sample.
        """
        # drug sensitivity
        selected_sample = self.drug_sensitivity_df.iloc[index]
        ic50_tensor = torch.tensor(
            [selected_sample['IC50']],
            dtype=self.drug_sensitivity_dtype,
            device=self.device
        )
        # SMILES
        token_indexes_tensor = self.smiles_dataset[
            self.smiles_dataset.sample_to_index_mapping[
                selected_sample['drug']
            ]
        ]  # yapf: disable
        # gene_expression
        gene_expression_tensor = self.gene_expression_dataset[
            self.gene_expression_dataset.sample_to_index_mapping[
                selected_sample['cell_line']]]
        return token_indexes_tensor, gene_expression_tensor, ic50_tensor
