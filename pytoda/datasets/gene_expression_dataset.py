"""GeneExpressionDataset module."""
import torch
from torch.utils.data import Dataset
from ._table_eager_dataset import _TableEagerDataset
from ._table_lazy_dataset import _TableLazyDataset
from ..types import FileList, GeneList

TABLE_DATASET_IMPLEMENTATIONS = {
    'eager': _TableEagerDataset,
    'lazy': _TableLazyDataset
}


class GeneExpressionDataset:
    """
    Gene expression dataset implementation.
    """

    def __init__(
        self,
        *gene_expression_filepaths: FileList,
        gene_list: GeneList = None,
        standardize: bool = True,
        min_max: bool = False,
        processing_parameters: dict = {},
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu'),
        backend: str = 'eager',
        **kwargs
    ) -> None:
        """
        Initialize a gene expression dataset.

        Args:
            gene_expression_filepaths (FileList): paths to .csv files.
                Currently, the only supported format is .csv, with gene
                profiles on rows and gene names as columns.
            gene_list (GeneList): a list of genes. Defaults to None.
            standardize (bool): perform data standardization. Defaults to True.
            min_max (bool): perform min-max scaling. Defaults to False.
            processing_parameters (dict): processing parameters.
                Defaults to {}.
            dtype (torch.dtype): data type. Defaults to torch.float.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
            backend (str): memeory management backend.
                Defaults to eager, prefer speed over memory consumption.
            kwargs (dict): additional parameters for pd.read_csv.
        """
        Dataset.__init__(self)
        if not (backend in TABLE_DATASET_IMPLEMENTATIONS):
            raise RuntimeError(
                'backend={} not supported! '.format(backend) +
                'Select one in [{}]'.
                format(','.join(TABLE_DATASET_IMPLEMENTATIONS.keys()))
            )
        self._dataset = TABLE_DATASET_IMPLEMENTATIONS[backend](
            filepaths=gene_expression_filepaths,
            feature_list=gene_list,
            standardize=standardize,
            min_max=min_max,
            processing_parameters=processing_parameters,
            dtype=dtype,
            device=device,
            **kwargs
        )
        self.gene_list = self._dataset.feature_list
        self.number_of_features = len(self.gene_list)
        self.max = self._dataset.max
        self.min = self._dataset.min
        self.mean = self._dataset.mean
        self.std = self._dataset.std
        self.sample_to_index_mapping = self._dataset.sample_to_index_mapping
        self.processing = self._dataset.processing

    def __len__(self) -> int:
        """Total number of samples."""
        return len(self._dataset)

    def __getitem__(self, index: int) -> torch.tensor:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            torch.tensor: a torch tensor of token indexes,
                for the current sample.
        """
        return self._dataset[index]
