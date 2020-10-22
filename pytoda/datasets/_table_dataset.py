"""Implementation of _TableDataset."""
import copy
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

from ..types import ArrayLike01D, FeatureList, Files, Tensor
from ._csv_eager_dataset import _CsvEagerDataset
from ._csv_lazy_dataset import _CsvLazyDataset
from ._csv_statistics import reduce_csv_statistics
from .base_dataset import DatasetDelegator
from .utils import concatenate_file_based_datasets


class _TableDataset(DatasetDelegator):
    """
    Table dataset abstract definition.

    The implementation is abstract and can be extend to define different
    data loading policies.
    """

    def __init__(
        self,
        filepaths: Files,
        feature_list: FeatureList = None,
        standardize: bool = True,
        min_max: bool = False,
        processing_parameters: dict = {},
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu'),
        chunk_size: int = 10000,
        **kwargs
    ) -> None:
        """
        Initialize a table dataset.

        Args:
            filepaths (Files): paths to .csv files.
            feature_list (GeneList): a list of features. Defaults to None.
            standardize (bool): perform data standardization. Defaults to True.
            min_max (bool): perform min-max scaling. Defaults to False.
            processing_parameters (dict): processing parameters.
                Defaults to {}.
            dtype (torch.dtype): data type. Defaults to torch.float.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
            chunk_size (int): size of the chunks in case of lazy reading, is
                ignored with 'eager' backend. Defaults to 10000.
            kwargs (dict): additional parameters for pd.read_csv.
        """
        self.processing = {}
        self.filepaths = filepaths
        self.feature_list = feature_list
        self.standardize = standardize
        self.min_max = min_max
        self.processing_parameters = processing_parameters
        self.dtype = dtype
        self.device = device
        self.chunk_size = chunk_size
        self.kwargs = copy.deepcopy(kwargs)
        if self.standardize and self.min_max:
            raise RuntimeError('Cannot both standardize and min-max scale')
        self.dataset = None
        self.max = None
        self.min = None
        self.mean = None
        self.std = None
        # NOTE: the dataset will be initialized and
        # designed to return numpy arrays,
        # the statistics will be updated accordingly
        self._setup_dataset()
        # NOTE: reduce statistics
        (  # yapf:disable
            self.feature_list, self.max, self.min, self.mean, self.std
        ) = reduce_csv_statistics(
            self.dataset.datasets, self.feature_list
        )

        # NOTE: adapt feature list, mapping and function
        self.feature_mapping = pd.Series(
            OrderedDict(
                [
                    (feature, index)
                    for index, feature in enumerate(self.feature_list)
                ]
            )
        )
        self.feature_fn = lambda df: df[self.feature_list]
        self.number_of_features = len(self.feature_list)
        # NOTE: define the transformation
        self.transform_fn = lambda example: example
        if self.standardize:
            mean: ArrayLike01D = self.processing_parameters.get('mean', self.mean)
            std: ArrayLike01D = self.processing_parameters.get('std', self.std)
            # support scalars, sequences (also of length 1) and arrays
            mean = np.array(mean, dtype=float).squeeze()
            std = np.array(std, dtype=float).squeeze()
            self.transform_fn = lambda example: ((example - mean) / std)
            self.processing = {
                'processing': 'standardize',
                'parameters': {
                    'mean': mean.tolist(),
                    'std': std.tolist()
                }
            }
        elif self.min_max:
            minimum: ArrayLike01D = self.processing_parameters.get('min', self.min)
            maximum: ArrayLike01D = self.processing_parameters.get('max', self.max)
            # support scalars as well as sequences (also of length 1) and arrays
            minimum = np.array(minimum, dtype=float).squeeze()
            maximum = np.array(maximum, dtype=float).squeeze()
            # must support DataFrame (eager dataset) and ..Series/array (lazy dataset or whatever cache index returns)
            self.transform_fn = lambda example: (
                (example - minimum) / (maximum - minimum)
            )
            self.processing = {
                'processing': 'min_max',
                'parameters': {
                    'min': minimum.tolist(),
                    'max': maximum.tolist()
                }
            }
        # apply preprocessing
        self._preprocess_dataset()

    def _setup_dataset(self) -> None:
        """
        Setup KeyDataset assigned to self.dataset for delegation.
        Defines feature_mapping and fits statistics."""
        raise NotImplementedError

    def _preprocess_dataset(self) -> None:
        """Preprocess the dataset."""
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tensor:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            torch.tensor: a torch tensor of table values
                for the current sample.
        """
        return torch.tensor(
            self.dataset[index], dtype=self.dtype, device=self.device
        )


class _TableLazyDataset(_TableDataset):
    """
    Table dataset using lazy loading.

    Suggested when handling datasets that can fit in the device memory.
    In case of datasets fitting in device memory consider using
    _TableEagerDataset for better performance.
    """

    def _setup_dataset(self) -> None:
        """Setup KeyDataset assigned to self.dataset for delegation."""
        self.dataset = concatenate_file_based_datasets(
            filepaths=self.filepaths,
            dataset_class=_CsvLazyDataset,
            feature_list=self.feature_list,
            chunk_size=self.chunk_size,
            **self.kwargs
        )

    def _preprocess_dataset(self) -> None:
        """Preprocess the dataset."""
        self.feature_fn = lambda sample: sample[self.feature_mapping[
            self.feature_list].values]
        for dataset in self.dataset.datasets:
            for index in dataset.cache:
                dataset.cache[index] = self.transform_fn(
                    self.feature_fn(dataset.cache[index])
                )


class _TableEagerDataset(_TableDataset):
    """
    Table dataset using eager loading.

    Suggested when handling datasets that can fit in the device memory.
    In case of out of memory errors consider using _TableLazyDataset.
    """

    def _setup_dataset(self) -> None:
        """Setup KeyDataset assigned to self.dataset for delegation."""
        self.dataset = concatenate_file_based_datasets(
            filepaths=self.filepaths,
            dataset_class=_CsvEagerDataset,
            feature_list=self.feature_list,
            **self.kwargs
        )

    def _preprocess_dataset(self) -> None:
        """Preprocess the dataset."""
        self.feature_fn = lambda sample: sample[self.feature_list]
        for dataset in self.dataset.datasets:
            dataset.df = self.transform_fn(self.feature_fn(dataset.df))
