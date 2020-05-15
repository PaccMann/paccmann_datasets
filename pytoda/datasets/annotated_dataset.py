"""Implementation of AnnotatedDataset class."""
import torch
import pandas as pd
from .base_dataset import IndexedDataset
from .dataframe_dataset import DataFrameDataset
from ..types import AnnotatedData, Union, List, Hashable


class AnnotatedDataset(DataFrameDataset):
    """
    Annotated samples in order of annotations csv, fetching data
    from passed dataset.
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
        self.datasource = dataset

        # processing of the dataframe for dataset setup
        df = pd.read_csv(
            self.annotations_filepath, **kwargs
        )
        columns = df.columns
        # handle annotation index
        if isinstance(annotation_index, int):
            self.annotation_index = columns[annotation_index]
        elif isinstance(annotation_index, str):
            self.annotation_index = annotation_index
        else:
            raise RuntimeError('annotation_index should be int or str.')
        # set the index explicitly
        df = df.set_index(
            self.annotation_index
        )
        DataFrameDataset.__init__(self, df)

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
        # get the number of labels
        self.number_of_tasks = len(self.labels)

    def __getitem__(self, index: int) -> AnnotatedData:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            AnnotatedData: a tuple containing two torch.tensors,
                representing respectively: compound token indexes and labels for
                the current sample.
        """
        # sample selection
        selected_sample = self.df.iloc[index]
        # label
        labels_tensor = torch.tensor(
            list(selected_sample[self.labels].values),  # base_dataset: why selecting self.labels here and not for all in __init__? want to change self.labels on the instance?
            dtype=torch.float,
            device=self.device
        )
        # sample
        sample = self.datasource.get_item_from_key(selected_sample.name)
        return sample, labels_tensor

    def get_item_from_key(self, key: Hashable) -> AnnotatedData:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            AnnotatedData: a tuple containing two torch.tensors,
                representing respectively: compound token indexes and labels for
                the current sample.
        """
        # sample selection
        selected_sample = self.df.loc[key, :]
        # label
        labels_tensor = torch.tensor(
            list(selected_sample[self.labels].values),
            dtype=torch.float,
            device=self.device
        )
        # sample
        sample = self.datasource.get_item_from_key(key)
        return sample, labels_tensor
