"""Implementation of _TableDataset."""
import torch
import copy
import pandas as pd
from collections import OrderedDict
from torch.utils.data import Dataset
from ._csv_dataset import reduce_csv_dataset_statistics
from ..types import FileList, FeatureList


class _TableDataset(Dataset):
    """
    Table dataset abstract definition.

    The implementation is abstract and can be extend to define different
    data loading policies.
    """

    def __init__(
        self,
        filepaths: FileList,
        feature_list: FeatureList = None,
        standardize: bool = True,
        min_max: bool = False,
        processing_parameters: dict = {},
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu'),
        **kwargs
    ) -> None:
        """
        Initialize a table dataset.

        Args:
            filepaths (FileList): paths to .csv files.
            feature_list (GeneList): a list of features. Defaults to None.
            standardize (bool): perform data standardization. Defaults to True.
            min_max (bool): perform min-max scaling. Defaults to False.
            processing_parameters (dict): processing parameters.
                Defaults to {}.
            dtype (torch.dtype): data type. Defaults to torch.float.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
            kwargs (dict): additional parameters for pd.read_csv.
        """
        Dataset.__init__(self)
        self.processing = {}
        self.filepaths = filepaths
        self.feature_list = feature_list
        if self.feature_list is not None:
            # NOTE: important to guarantee feature order preservation when
            # multiple datasets are passed
            self.feature_ordering = {
                feature: index
                for index, feature in enumerate(self.feature_list)
            }
        else:
            self.feature_ordering = None
        self.standardize = standardize
        self.min_max = min_max
        self.processing_parameters = processing_parameters
        self.dtype = dtype
        self.device = device
        self.kwargs = copy.deepcopy(kwargs)
        if self.standardize and self.min_max:
            raise RuntimeError('Cannot both standardize and min-max scale')
        self._dataset = None
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
        ) = reduce_csv_dataset_statistics(
            self._dataset.datasets, self.feature_list, self.feature_ordering
        )
        # NOTE: recover sample and index mappings
        self.sample_to_index_mapping = {}
        self.index_to_sample_mapping = {}
        for index in range(len(self._dataset)):
            dataset_index, sample_index = self._dataset.get_index_pair(index)
            dataset = self._dataset.datasets[dataset_index]
            sample = dataset.index_to_sample_mapping[sample_index]
            self.sample_to_index_mapping[sample] = index
            self.index_to_sample_mapping[index] = sample
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
            mean = self.processing_parameters.get('mean', self.mean)
            std = self.processing_parameters.get('std', self.std)
            self.transform_fn = lambda example: ((example - mean) / std)
            self.processing = {
                'processing': 'standardize',
                'parameters': {
                    'mean': list(mean),
                    'std': list(std)
                }
            }
        elif self.min_max:
            minimum = self.processing_parameters.get('min', self.min)
            maximum = self.processing_parameters.get('max', self.max)
            self.transform_fn = lambda example: (
                (example - minimum) / float(maximum - minimum)
            )
            self.processing = {
                'processing': 'min_max',
                'parameters': {
                    'min': minimum,
                    'max': maximum
                }
            }
        # apply preprocessing
        self._preprocess_dataset()

    def _setup_dataset(self) -> None:
        """Setup the dataset."""
        raise NotImplementedError

    def _preprocess_dataset(self) -> None:
        """Preprocess the dataset."""
        raise NotImplementedError

    def __len__(self) -> int:
        "Total number of samples."
        return len(self._dataset)

    def __getitem__(self, index: int) -> torch.tensor:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            torch.tensor: a torch tensor of table values
                for the current sample.
        """
        return torch.tensor(
            self._dataset[index], dtype=self.dtype, device=self.device
        )
