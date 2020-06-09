"""Implementation of _SMILESDataset."""
import logging

from ..types import FileList
from ._smi_eager_dataset import _SmiEagerDataset
from ._smi_lazy_dataset import _SmiLazyDataset
from .base_dataset import DatasetDelegator
from .utils import concatenate_file_based_datasets

logger = logging.getLogger(__name__)


SMILES_DATASET_IMPLEMENTATIONS = {  # get class and acceptable keywords
    'eager': (_SmiEagerDataset, {'name'}),
    'lazy': (_SmiLazyDataset, {'chunk_size', 'name'}),
}


class SMILESDataset(DatasetDelegator):
    """
    SMILES dataset abstract definition.

    The implementation is abstract and can be extend to define different
    data loading policies.
    """

    def __init__(
        self,
        *smi_filepaths: FileList,
        backend: str = 'eager',
        name: str = 'smiles-dataset',
        **kwargs
    ) -> None:
        """
        Initialize a SMILES dataset.

        Args:
            smi_filepaths (FileList): paths to .smi files.
            name (str): name of the SMILESDataset.
            backend (str): memory management backend.
                Defaults to eager, prefer speed over memory consumption.
            kwargs (dict): additional arguments for dataset constructor.

        TODO
        NOTE: If the setup is too slow, consider skipping all SMILES language
            transforms. To achieve this, set ALL following arguments to False:
                - `canonical`
                - `augment`
                - `kekulize`
                - `all_bonds_explicit`
                - `all_hs_explicit`
                - `remove_bonddir`
                - `remove_chirality`
        """
        # Parse language object and data paths
        self.smi_filepaths = smi_filepaths
        self.backend = backend
        self.name = name

        dataset_class, valid_keys = SMILES_DATASET_IMPLEMENTATIONS[
            self.backend
        ]
        kwargs = dict(
            (k, v) for k, v in self.kwargs.items() if k in valid_keys
        )

        self.dataset = concatenate_file_based_datasets(
            filepaths=self.smi_filepaths,
            dataset_class=dataset_class,
            **kwargs
        )

        DatasetDelegator.__init__(self)  # delegate to self.dataset
        if self.has_duplicate_keys:
            raise KeyError('Please remove duplicates from your .smi file.')
