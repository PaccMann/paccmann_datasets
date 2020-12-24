"""_CacheDataset abstract implementation."""
import tempfile
import shutil
import logging
import diskcache as dc
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class _CacheDataset(Dataset):
    """
    Dataset supporting a disk cache.

    Suggested when handling datasets that can't fit in the device memory.
    """

    def __init__(self) -> None:
        """Initialize a dataset using a disk cache."""
        Dataset.__init__(self)
        self.cache_filepath = tempfile.mkdtemp()
        self.cache = dc.Cache(self.cache_filepath)

    def __del__(self):
        """Delete the _CacheDataset."""
        try:
            shutil.rmtree(self.cache_filepath)
        except Exception:
            logger.warning('_CacheDataset cache deletion not performed!')
