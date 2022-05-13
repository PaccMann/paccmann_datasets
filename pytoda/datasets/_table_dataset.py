"""Implementation of _TableDataset."""
import copy
from functools import partial

import numpy as np
import torch

from pytoda.warnings import device_warning

from ..types import CsvSourceData, FeatureList, Files, Optional, Tensor
from ._csv_eager_dataset import _CsvEagerDataset
from ._csv_lazy_dataset import _CsvLazyDataset
from ._csv_statistics import reduce_csv_statistics
from .base_dataset import DatasetDelegator
from .utils import concatenate_file_based_datasets


# https://github.com/scikit-learn/scikit-learn/blob/0fb307bf39bbdacd6ed713c00724f8f871d60370/sklearn/preprocessing/_data.py#L63
def _handle_zeros_in_scale(scale, copy=True):
    """
    This method is copied `from sklearn.preprocessing._data`
    Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.
    """
    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == 0.0:
            scale = 1.0
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


def transform_not(data: CsvSourceData):
    return data


def transform_standardize(data: CsvSourceData, mean: np.ndarray, std: np.ndarray):
    return (data - mean) / _handle_zeros_in_scale(std, copy=False)


def transform_minmax(data: CsvSourceData, minimum: np.ndarray, maximum: np.ndarray):
    return (data - minimum) / _handle_zeros_in_scale(maximum - minimum, copy=False)


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
        impute: Optional[float] = None,
        dtype: torch.dtype = torch.float,
        chunk_size: int = 10000,
        device: torch.device = None,
        **kwargs,
    ) -> None:
        """
        Initialize a table dataset.

        Args:
            filepaths (Files): paths to .csv files.
            feature_list (GeneList): a list of features. Defaults to None.
            standardize (bool): perform data standardization. Defaults to True.
            min_max (bool): perform min-max scaling. Defaults to False.
            processing_parameters (dict): processing parameters.
                Keys can be 'min', 'max' or 'mean', 'std'
                respectively. Values must be readable by `np.array`, and the
                required order and subset of features has to match that
                determined by the dataset setup (see `self.feature_list` after
                initialization). Defaults to {}.
            impute (Optional[float]): NaN imputation with value if
                given. Defaults to None.
            dtype (torch.dtype): data type. Defaults to torch.float.
            chunk_size (int): size of the chunks in case of lazy reading, is
                ignored with 'eager' backend. Defaults to 10000.
            device (torch.device): DEPRECATED
            kwargs (dict): additional parameters for pd.read_csv.
        """
        self.filepaths = filepaths
        self.initial_feature_list = feature_list
        self.standardize = standardize
        self.min_max = min_max
        self.processing_parameters = processing_parameters
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.kwargs = copy.deepcopy(kwargs)
        if self.standardize and self.min_max:
            raise RuntimeError('Cannot both standardize and min-max scale')
        self.dataset = None
        self.max = None
        self.min = None
        self.mean = None
        self.std = None
        device_warning(device)

        # the dataset(s) will be initialized individually,
        # the collected statistics will later be updated and finally applied
        # TODO: depending on processing_parameters and standardize/minmax,
        # statistics need not be computed on setup.
        self._setup_dataset()

        # reduce statistics and find definitive feature_list
        (
            self.feature_list,
            self.max,
            self.min,
            self.mean,
            self.std,
        ) = reduce_csv_statistics(self.dataset.datasets, self.initial_feature_list)
        self.number_of_features = len(self.feature_list)

        # given the statistics, we define the transformation
        self.transform_fn = transform_not
        self.processing = {}
        if self.standardize:
            mean = self.processing_parameters.get('mean', self.mean)
            std = self.processing_parameters.get('std', self.std)
            # support scalars, sequences (also of length 1) and arrays
            mean = np.array(mean, dtype=float).squeeze()
            std = np.array(std, dtype=float).squeeze()
            self.transform_fn = partial(transform_standardize, mean=mean, std=std)
            self.processing = {
                'processing': 'standardize',
                'parameters': {'mean': mean.tolist(), 'std': std.tolist()},
            }
        elif self.min_max:
            minimum = self.processing_parameters.get('min', self.min)
            maximum = self.processing_parameters.get('max', self.max)
            # support scalars, sequences (also of length 1) and arrays
            minimum = np.array(minimum, dtype=float).squeeze()
            maximum = np.array(maximum, dtype=float).squeeze()
            self.transform_fn = partial(
                transform_minmax, minimum=minimum, maximum=maximum
            )
            self.processing = {
                'processing': 'min_max',
                'parameters': {'min': minimum.tolist(), 'max': maximum.tolist()},
            }
        # Filter, order and transform the datasets and impute missing values
        for dataset in self.dataset.datasets:
            dataset.transform_dataset(
                transform_fn=self.transform_fn,
                feature_list=self.feature_list,
                impute=impute,
            )

    def _setup_dataset(self) -> None:
        """
        Setup KeyDataset assigned to self.dataset for delegation.
        Defines feature_mapping and fits statistics."""
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
        return torch.tensor(self.dataset[index], dtype=self.dtype)


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
            feature_list=self.initial_feature_list,
            chunk_size=self.chunk_size,
            **self.kwargs,
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
            feature_list=self.initial_feature_list,
            **self.kwargs,
        )
