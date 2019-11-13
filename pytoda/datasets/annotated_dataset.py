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
        dataset: Dataset,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu'),
        **kwargs
    ) -> None:
        """
        Initialize an annotated dataset.
        E.g. the  dataset could be SMILES and the annotations could be
        single or multi task labels.

        Args:
            annotations_filepath (str): path to the annotations of a dataset
                .csv file. Currently, the only supported format is .csv, the
                last column should point to an ID that is also contained in
                dataset.
            dataset (Dataset): path to .smi file.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
            kwargs (dict): additional parameter for pd.read_csv. E.g. index_col
                defaults to 0 (set in the constructor).
        """
        Dataset.__init__(self)

        self.device = device
        kwargs['index_col'] = kwargs.get('index_col', 0)

        self.annotations_filepath = annotations_filepath
        self.dataset = dataset

        self.annotated_data_df = pd.read_csv(
            self.annotations_filepath, **kwargs
        )

        # Multilabel classification case
        self.num_tasks = len(self.annotated_data_df.columns) - 1
        self.id_column_name = self.annotated_data_df.columns[-1]

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
        token_indexes_tensor = self.dataset[
            self.dataset.sample_to_index_mapping[
                selected_sample[self.id_column_name]
            ]
        ]   # yapf: disable
        return token_indexes_tensor, labels_tensor
