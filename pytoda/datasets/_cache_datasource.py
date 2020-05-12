"""_CacheDatasource abstract implementation."""
import tempfile
import shutil
import diskcache as dc


class _CacheDatasource:
    """
    Supporting Datasets with underlying disk cache.

    Suggested when handling datasets that can't fit in the device memory.
    """

    def __init__(self) -> None:
        """Initialize a dataset using a disk cache."""
        self.cache_filepath = tempfile.mkdtemp()
        self.cache = dc.Cache(self.cache_filepath)

    def __del__(self):
        """Delete the _CacheDatasource."""
        shutil.rmtree(self.cache_filepath)
