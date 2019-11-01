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
        smiles_language: SMILESLanguage = None,
        padding: bool = True,
        padding_length: int = None,
        add_start_and_stop: bool = False,
        augment: bool = False,
        gene_list: GeneList = None,
        gene_expression_standardize: bool = True,
        gene_expression_dtype: torch.dtype = torch.float,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu'),
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
            smiles_language (SMILESLanguage): a smiles language.
                Defaults to None.
            padding (bool): pad sequences to longest in the smiles language.
                Defaults to True.
            padding_length (int): manually sets number of applied paddings,
                applies only if padding is True. Defaults to None.
            add_start_and_stop (bool): add start and stop token indexes.
                Defaults to False.
            augment (bool): perform SMILES augmentation. Defaults to False.
            gene_list (GeneList): a list of genes.
            gene_expression_standardize (bool): perform gene expression
                data standardization. Defaults to True.
            gene_expression_dtype (torch.dtype): gene expression data type.
                Defaults to torch.float.
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
            device=self.device,
            backend=self.backend
        )
        # gene expression
        self.gene_expression_dataset = GeneExpressionDataset(
            self.gene_expression_filepath,
            gene_list=gene_list,
            standardize=gene_expression_standardize,
            dtype=gene_expression_dtype,
            device=self.device,
            backend=self.backend,
            index_col=0
        )
        # drug sensitivity
        self.drug_sensitivity_dtype = drug_sensitivity_dtype
        self.drug_sensitivity_min_max = drug_sensitivity_min_max
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
            self.drug_sensitivity_minimum = (
                self.drug_sensitivity_df['IC50'].min()
            )
            self.drug_sensitivity_maximum = (
                self.drug_sensitivity_df['IC50'].max()
            )
            self.drug_sensitivity_df['IC50'] = (
                (
                    self.drug_sensitivity_df['IC50'] -
                    self.drug_sensitivity_minimum
                ) / (
                    self.drug_sensitivity_maximum -
                    self.drug_sensitivity_minimum
                )
            )

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
            self.smiles_dataset.sample_to_index_mapping[selected_sample['drug']
                                                        ]]
        # gene_expression
        gene_expression_tensor = self.gene_expression_dataset[
            self.gene_expression_dataset.sample_to_index_mapping[
                selected_sample['cell_line']]]
        return token_indexes_tensor, gene_expression_tensor, ic50_tensor
