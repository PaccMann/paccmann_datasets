"""GeneExpressionDataset module."""
import torch
from .base_dataset import DatasetDelegator
from ._table_dataset import _TableEagerDataset, _TableLazyDataset
from ..types import FileList, GeneList

TABLE_DATASET_IMPLEMENTATIONS = {
    'eager': _TableEagerDataset,
    'lazy': _TableLazyDataset
}


class GeneExpressionDataset(DatasetDelegator):
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
        chunk_size: int = 10000,
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
            backend (str): memory management backend.
                Defaults to eager, prefer speed over memory consumption.
            chunk_size (int): size of the chunks in case of lazy reading, is
                ignored with 'eager' backend. Defaults to 10000.
            kwargs (dict): additional parameters for pd.read_csv.
        """
        if not (backend in TABLE_DATASET_IMPLEMENTATIONS):
            raise RuntimeError(
                'backend={} not supported! '.format(backend) +
                'Select one in [{}]'.
                format(','.join(TABLE_DATASET_IMPLEMENTATIONS.keys()))
            )
        self.dataset = TABLE_DATASET_IMPLEMENTATIONS[backend](
            filepaths=gene_expression_filepaths,
            feature_list=gene_list,
            standardize=standardize,
            min_max=min_max,
            processing_parameters=processing_parameters,
            dtype=dtype,
            device=device,
            chunk_size=chunk_size,
            **kwargs
        )
        self.gene_list = self.dataset.feature_list
        self.number_of_features = len(self.gene_list)
        DatasetDelegator.__init__(self)  # delegate to self.dataset
