"""_CacheDataset abstract implementation."""
import tempfile
import shutil
import diskcache as dc
from torch.utils.data import Dataset


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
        shutil.rmtree(self.cache_filepath)
