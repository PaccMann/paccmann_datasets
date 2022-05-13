"""Implementation of AnnotatedDataset class."""
import pandas as pd
import torch

from pytoda.warnings import device_warning

from ..types import AnnotatedData, Hashable, List, Union
from .base_dataset import AnyBaseDataset
from .dataframe_dataset import DataFrameDataset


class AnnotatedDataset(DataFrameDataset):
    """
    Annotated samples in order of annotations csv, fetching data
    from passed dataset.
    """

    def __init__(
        self,
        annotations_filepath: str,
        dataset: AnyBaseDataset,
        annotation_index: Union[int, str] = -1,
        label_columns: Union[List[int], List[str]] = None,
        dtype: torch.dtype = torch.float,
        device: torch.device = None,
        **kwargs,
    ) -> None:
        """
        Initialize an annotated dataset via additional annotations dataframe.
        E.g. the  dataset could be SMILES and the annotations could be
        single or multi task labels.

        Args:
            annotations_filepath (str): path to the annotations of a dataset.
                Currently, the supported formats are column separated files.
                The default structure assumes that the last column contains an
                id that is also used in the dataset provided.
            dataset (AnyBaseDataset): instance of a AnyBaseDataset (supporting
                key lookup API of KeyDataset), e.g. a SMILESDataset.
            annotation_index (Union[int, str]): positional or string for the
                column containing the annotation index of keys to get items in
                the passed dataset. Defaults to -1, i.e. the last column.
            label_columns (Union[List[int], List[str]]): indexes (positional
                or strings) for the annotations. Defaults to None, a.k.a. all
                the columns, except the annotation index, are considered
                annotation labels.
            dtype (torch.dtype): torch data type for labels. Defaults to
                torch.float.
            device (torch.device): DEPRECATED
            kwargs (dict): additional parameter for pd.read_csv.
        """
        self.annotations_filepath = annotations_filepath
        self.datasource = dataset
        self.dtype = dtype
        device_warning(device)

        # processing of the dataframe for dataset setup
        df = pd.read_csv(self.annotations_filepath, **kwargs)
        columns = df.columns
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
        # get the number of labels
        self.number_of_tasks = len(self.labels)

        # set the index explicitly, and discard non label columns
        df = df.set_index(self.annotation_index)[self.labels]
        DataFrameDataset.__init__(self, df)

    def __getitem__(self, index: int) -> AnnotatedData:
        """
        Get key from integer index.

        Args:
            index (int): index annotations to get key and annotation labels,
                where the key is used to fetch the item.

        Returns:
            AnnotatedData: a tuple containing the item itself (with type
                depending on passed dataset) and a torch.Tensor of
                labels for the current item.
        """
        # sample selection
        selected_sample = self.df.iloc[index]
        return self._make_return_tuple(selected_sample)

    def get_item_from_key(self, key: Hashable) -> AnnotatedData:
        """
        Get item via key.

        Args:
            key (Hashable): key of the item and annotations to fetch.

        Returns:
            AnnotatedData: a tuple containing the item itself (with type
                depending on passed dataset) and a torch.Tensor of
                labels for the current item.
        """
        # sample selection
        selected_sample = self.df.loc[key, :]
        return self._make_return_tuple(selected_sample)

    def _make_return_tuple(self, lables_series: pd.Series) -> AnnotatedData:
        # sample
        sample = self.datasource.get_item_from_key(lables_series.name)
        # label
        labels_tensor = torch.tensor(list(lables_series.values), dtype=self.dtype)
        return sample, labels_tensor
