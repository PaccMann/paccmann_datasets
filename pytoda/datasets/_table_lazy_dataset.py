"""Implementation of _TableLazyDataset."""
import torch
from .utils import concatenate_file_based_datasets
from ..types import FileList, FeatureList
from ._table_dataset import _TableDataset
from ._csv_lazy_dataset import _CsvLazyDataset
from ._csv_dataset import reduce_csv_dataset_statistics


class _TableLazyDataset(_TableDataset):
    """
    Table dataset using lazy loading.

    Suggested when handling datasets that can fit in the device memory.
    In case of datasets fitting in device memory consider using
    _TableEagerDataset for better performance.
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
        chunk_size: int = 10000,
        **kwargs
    ) -> None:
        """
        Initialize a table lazy dataset.

        Args:
            filepaths (FileList): paths to .csv files.
            feature_list (GeneList): a list of features. Defaults to None.
            standardize (bool): perform data standardization. Defaults to True.
            min_max (bool): perform data normalization. Defaults to False.
            processing_parameters (dict): processing parameters.
                Defaults to None.
            dtype (torch.dtype): data type. Defaults to torch.float.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
            chunk_size (int): size of the chunks. Defauls to 10000.
            kwargs (dict): additional parameters for pd.read_csv.
        """
        self.chunk_size = chunk_size
        super(_TableLazyDataset, self).__init__(
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
            dataset_class=_CsvLazyDataset,
            feature_list=self.feature_list,
            chunk_size=self.chunk_size,
            **self.kwargs
        )

    def _preprocess_dataset(self) -> None:
        """Preprocess the dataset."""
        self.feature_fn = lambda sample: sample[self.feature_mapping[
            self.feature_list].values]
        for dataset in self._dataset.datasets:
            for index in dataset.cache:
                dataset.cache[index] = self.transform_fn(
                    self.feature_fn(dataset.cache[index])
                )
