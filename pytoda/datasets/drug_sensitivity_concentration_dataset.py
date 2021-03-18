"""Implementation of DrugSensitivityConcentrationDataset."""
from typing import Callable

import numpy as np
import torch

from ..smiles.smiles_language import SMILESTokenizer
from ..types import DrugSensitivityConcentrationData, GeneList, Tuple
from .drug_sensitivity_dataset import DrugSensitivityDataset


class DrugSensitivityConcentrationDataset(DrugSensitivityDataset):
    """
    Drug sensitivity concentration dataset implementation.
    """

    def __init__(
        self,
        drug_sensitivity_filepath: str,
        smi_filepath: str,
        gene_expression_filepath: str,
        smiles_language: SMILESTokenizer,
        column_names: Tuple[str] = ['drug', 'cell_line', 'concentration', 'viability'],
        concentration_transform: Callable[[float], float] = np.log10,
        iterate_dataset: bool = False,
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
        Initialize a drug sensitivity concentration dataset.

        Args:
            drug_sensitivity_filepath (str): path to drug sensitivity .csv file.
                Currently, the only supported format is .csv, with an index and three
                header columns named as specified in the column_names argument.
            smi_filepath (str): path to .smi file.
            gene_expression_filepath (str): path to gene expression .csv file.
                Currently, the only supported format is .csv, with an index and header
                columns containing gene names.
            smiles_language (SMILESTokenizer): a smiles language/tokenizer must be
                passed. Specifies tokens and all transforms for SMILES conversion.
            column_names (Tuple[str]): Names of columns in data files to retrieve
                molecules, cell-line-data, drug concentration and viability (label).
                Defaults to ['drug', 'cell_line', 'concentration', 'viability'].
                All but the 2nd last (concentration) are passed to
                drug_sensitivity_dataset.
            concentration_transform (Callable[[float], float]):  A callable to convert
                the raw concentration into an input for the model. E.g. if raw
                concentration is uMol, torch.log10 could make sense.
                Defaults to torch.log10.
                NOTE: To switch it off, pass `lambda x:x`.
            iterate_dataset (bool): whether to go through all SMILES in the
                dataset to extend/build vocab, find longest sequence, and
                checks the passed padding length if applicable. Defaults to
                False.
            gene_list (GeneList): a list of genes.
            gene_expression_standardize (bool): perform gene expression
                data standardization. Defaults to True.
            gene_expression_min_max (bool): perform min-max scaling on gene
                expression data. Defaults to False.
            gene_expression_processing_parameters (dict): transformation
                parameters for gene expression, e.g. for min-max scaling.
                Defaults to {}.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
            backend (str): memory management backend.
                Defaults to eager, prefer speed over memory consumption.
                Note that at the moment only the gene expression and the
                smiles datasets implement both backends. The drug sensitivity
                data are loaded in memory.
        """
        super().__init__(
            drug_sensitivity_filepath=drug_sensitivity_filepath,
            smi_filepath=smi_filepath,
            gene_expression_filepath=gene_expression_filepath,
            column_names=column_names[:2] + [column_names[-1]],
            drug_sensitivity_min_max=False,
            drug_sensitivity_processing_parameters={},
            smiles_language=smiles_language,
            iterate_dataset=iterate_dataset,
            gene_list=gene_list,
            gene_expression_standardize=gene_expression_standardize,
            gene_expression_min_max=gene_expression_min_max,
            gene_expression_processing_parameters=gene_expression_processing_parameters,
            device=device,
            backend=backend,
        )

        self.concentration_name = column_names[2]
        self.concentration_transform = concentration_transform

    def __getitem__(self, index: int) -> DrugSensitivityConcentrationData:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            DrugSensitivityConcentrationData: a tuple containing four torch.Tensors,
                representing respectively:
                - compound token indexes,
                - gene expression values,
                - drug concentration,
                - cell viability.
        """
        token_indexes, gene_expression, viability = super().__getitem__(index)

        concentration = torch.tensor(
            [
                self.concentration_transform(
                    self.drug_sensitivity_df.iloc[index][self.concentration_name]
                )
            ],
            dtype=self.drug_sensitivity_dtype,
            device=self.device,
        )

        return token_indexes, gene_expression, concentration, viability