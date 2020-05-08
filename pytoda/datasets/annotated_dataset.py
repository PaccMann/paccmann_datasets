"""Implementation of AnnotatedDataset class."""
import torch
import pandas as pd
from torch.utils.data import Dataset
from .base_dataset import IndexedDataset, DatasetDelegator
from ..types import AnnotatedData, Union, List


class AnnotatedDataset(DatasetDelegator):
    """
    Annotated dataset implementation.
    """

    def __init__(
        self,
        annotations_filepath: str,
        dataset: IndexedDataset,
        annotation_index: Union[int, str] = -1,
        label_columns: Union[List[int], List[str]] = None,
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu'),
        **kwargs
    ) -> None:  # TODO document string labels
        """
        Initialize an annotated dataset via additional annotations dataframe.
        E.g. the  dataset could be SMILES and the annotations could be
        single or multi task labels.

        Args:
            annotations_filepath (str): path to the annotations of a dataset.
                Currently, the supported formats are column separated files.
                The default structure assumes that the last column contains an
                id that is also used in the dataset provided.
            dataset (Dataset): instance of a IndexedDataset (supporting
                label indices). E.g. a SMILESDataset
            annotation_index (Union[int, str]): positional or string for the
                column containing the annotation index. Defaults to -1, a.k.a.
                the last column.
            label_columns (Union[List[int], List[str]]): indexes (positional
                or strings) for the annotations. Defaults to None, a.k.a. all
                the columns, except the annotation index, are considered
                annotation labels.
            dtype (torch.dtype): data type. Defaults to torch.float.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
            kwargs (dict): additional parameter for pd.read_csv.
        """
        self.device = device
        self.annotations_filepath = annotations_filepath
        self.dataset = dataset
        self.annotated_data_df = pd.read_csv(
            self.annotations_filepath, **kwargs
        )
        # post-processing of the dataframe
        columns = self.annotated_data_df.columns
        # handle annotation index
        if isinstance(annotation_index, int):
            self.annotation_index = columns[annotation_index]
        elif isinstance(annotation_index, str):
            self.annotation_index = annotation_index
        else:
            raise RuntimeError('annotation_index should be int or str.')
        # handle labels
        if label_columns is None:
            self.labels = [
                column for column in columns if column != self.annotation_index
            ]
        elif all([isinstance(column, int) for column in label_columns]):
            self.labels = columns[label_columns]
        elif all([isinstance(column, str) for column in label_columns]):
            self.labels = label_columns
        else:
            raise RuntimeError(
                'label_columns should be an iterable containing int or str'
            )
        # set the index explicitly
        self.annotated_data_df = self.annotated_data_df.set_index(
            self.annotation_index
        )
        # get the number of labels
        self.number_of_tasks = len(self.labels)

    def assert_matching_keys(self):  # base_dataset: use in tests
        e = ValueError('Annotation data index does not match dataset keys')
        try:
            if any(self.annotated_data_df.index != list(self.keys())):
                raise e
        except ValueError:  # length mismatch
            raise e

    def __len__(self) -> int:
        "Total number of samples."
        return len(self.annotated_data_df)  # base_dataset: any checks on synchronized data in dataset?

    def __getitem__(self, index: int) -> AnnotatedData:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            AnnotatedData: a tuple containing two torch.tensors,
                representing respetively: compound token indexes and labels for
                the current sample.
        """
        # sample selection
        selected_sample = self.annotated_data_df.iloc[index]
          # base_dataset: require
          # - matching order (as is, no guarantee)
          # - dataset.get_key in df.index (would raise)
          # - both, as in self.assert_matching_keys
        # label
        labels_tensor = torch.tensor(
            list(selected_sample[self.labels].values),
            dtype=torch.float,
            device=self.device
        )
        # sample
        sample = self.dataset[
            self.dataset.sample_to_index_mapping[selected_sample.name]
        ]   # yapf: disable
        return sample, labels_tensor
