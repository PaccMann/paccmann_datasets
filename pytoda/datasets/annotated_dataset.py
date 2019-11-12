"""Implementation of AnnotatedDataset class."""
import torch
import pandas as pd
from torch.utils.data import Dataset
from ..types import DrugSensitivityData


class AnnotatedDataset(Dataset):
    """
    Annotated dataset implementation.
    """

    def __init__(
        self,
        annotations_filepath: str,
        input_data: Dataset,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu'),
        backend: str = 'eager',
        **kwargs
    ) -> None:
        """
        Initialize an annotated dataset.
        E.g. the  samples could be SMILES and the annotations a single or
        multiple labels.

        Args:
            annotations_filepath (str): path to the annotations of a dataset
                .csv file. Currently, the only supported format is .csv,
                with an index and three header columns named: "drug",
                "cell_line", "IC50".
            data (Dataset): path to .smi file.
            smiles_langśage (SMILESLanguage): a smiles language.
                Defaults to None.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
            backend (str): memeory management backend.
                Defaults to eager, prefer speed over memory consumption.
                Note that at the moment only the gene expression and the
                smiles datasets implement both backends. The drug sensitivity
                data are loaded in memory.
            kwargs (dict): additional parameters for pd.read_csv.
        """
        Dataset.__init__(self)
        self.annotations_filepath = annotations_filepath
        # device
        self.device = device
        # backend
        self.backend = backend
        # e.g. SMILES
        self.input_data = input_data

        self.annotated_data_df = pd.read_csv(
            self.annotations_filepath, index_col=0, **kwargs
        )

        # Multilabel classification case
        self.num_tasks = len(self.annotated_data_df.columns) - 1

    def __len__(self) -> int:
        "Total number of samples."
        return len(self.annotated_data_df)

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
        # Labels
        selected_sample = self.annotated_data_df.iloc[index]
        labels_tensor = torch.tensor(
            list(selected_sample[:self.num_tasks].values),
            dtype=torch.float,
            device=self.device
        )
        # e.g. SMILES
        token_indexes_tensor = self.input_data[
            self.input_data.sample_to_index_mapping[selected_sample['mol_id']]
        ]   # yapf: disable
        return token_indexes_tensor, labels_tensor
