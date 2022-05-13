"""Implementation of DrugSensitivityDoseDataset."""
from typing import Callable

import numpy as np
import torch

from pytoda.warnings import device_warning

from ..smiles.smiles_language import SMILESTokenizer
from ..types import DrugSensitivityDoseData, GeneList, Tuple
from .drug_sensitivity_dataset import DrugSensitivityDataset


class DrugSensitivityDoseDataset(DrugSensitivityDataset):
    """
    Drug sensitivity dose dataset implementation.
    """

    def __init__(
        self,
        drug_sensitivity_filepath: str,
        smi_filepath: str,
        gene_expression_filepath: str,
        smiles_language: SMILESTokenizer,
        column_names: Tuple[str] = ['drug', 'cell_line', 'dose', 'viability'],
        dose_transform: Callable[[float], float] = np.log10,
        iterate_dataset: bool = False,
        gene_list: GeneList = None,
        gene_expression_standardize: bool = True,
        gene_expression_min_max: bool = False,
        gene_expression_processing_parameters: dict = {},
        gene_expression_dtype: torch.dtype = torch.float,
        gene_expression_kwargs: dict = {},
        backend: str = 'eager',
        device: torch.device = None,
        **kwargs,
    ) -> None:
        """
        Initialize a drug sensitivity dose dataset.

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
                molecules, cell-line-data, drug dose and viability (label).
                Defaults to ['drug', 'cell_line', 'dose', 'viability'].
                All but the 2nd last (dosedose) are passed to
                drug_sensitivity_dataset.
            dose_transform (Callable[[float], float]):  A callable to convert
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
            backend (str): memory management backend.
                Defaults to eager, prefer speed over memory consumption.
                Note that at the moment only the gene expression and the
                smiles datasets implement both backends. The drug sensitivity
                data are loaded in memory.
            device (torch.device): DEPRECATED
            **kwargs: Additional keyword arguments for parent class
                (DrugSensitivityDataset).
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
            backend=backend,
            **kwargs,
        )

        self.dose_name = column_names[2]
        self.dose_transform = dose_transform
        device_warning(device)

    def __getitem__(self, index: int) -> DrugSensitivityDoseData:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            DrugSensitivityDoseDataset: a tuple containing four torch.Tensors,
                representing respectively:
                - compound token indexes,
                - gene expression values,
                - drug concentration,
                - cell viability.
        """
        token_indexes, gene_expression, viability = super().__getitem__(index)

        dose = torch.tensor(
            [self.dose_transform(self.drug_sensitivity_df.iloc[index][self.dose_name])],
            dtype=self.drug_sensitivity_dtype,
        )

        return token_indexes, gene_expression, dose, viability
