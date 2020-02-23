"""Implementation of _TableEagerDataset."""
import torch
from .utils import concatenate_file_based_datasets
from ..types import FileList, FeatureList
from ._table_dataset import _TableDataset
from ._csv_eager_dataset import _CsvEagerDataset


class _TableEagerDataset(_TableDataset):
    """
    Table dataset using eager loading.

    Suggested when handling datasets that can fit in the device memory.
    In case of out of memory errors consider using _TableLazyDataset.
    """

    def __init__(
        self,
        filepaths: FileList,
        feature_list: FeatureList = None,
        standardize: bool = True,
        min_max: bool = False,
        processing_parameters: dict = None,
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu'),
        **kwargs
    ) -> None:
        """
        Initialize a table eager dataset.

        Args:
            filepaths (FileList): paths to .csv files.
            feature_list (GeneList): a list of features. Defaults to None.
            standardize (bool): perform data standardization. Defaults to True.
            min_max (bool): perform min-max scaling. Defaults to False.
            processing_parameters (dict): processing parameters.
                Defaults to None.
            dtype (torch.dtype): data type. Defaults to torch.float.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
            kwargs (dict): additional parameters for pd.read_csv.
        """
        super(_TableEagerDataset, self).__init__(
            filepaths=filepaths,
            feature_list=feature_list,
            standardize=standardize,
            min_max=min_max,
            processing_parameters=processing_parameters,
            dtype=dtype,
            device=device,
            **kwargs
        )

    def _setup_dataset(self) -> None:
        """Setup the dataset."""
        self._dataset = concatenate_file_based_datasets(
            filepaths=self.filepaths,
            dataset_class=_CsvEagerDataset,
            feature_list=self.feature_list,
            dtype={'cell_line': str},
            **self.kwargs
        )

    def _preprocess_dataset(self) -> None:
        """Preprocess the dataset."""
        self.feature_fn = lambda sample: sample[self.feature_list]
        for dataset in self._dataset.datasets:
            dataset.df = self.transform_fn(self.feature_fn(dataset.df))
