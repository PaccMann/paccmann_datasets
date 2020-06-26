"""_CacheDatasource abstract implementation."""
import os
import shutil
import tempfile
import warnings

import diskcache as dc


def sizeof_fmt(num, suffix='B'):
    """Source: https://stackoverflow.com/a/1094933"""
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


class _CacheDatasource:
    """
    Supporting Datasets with underlying disk cache.

    Suggested when handling datasets that can't fit in the device memory.
    """
    size_limit = 1073741824  # default limit of 1GiB from diskcash

    def __init__(self, size_limit=None, fit_size_limit_filepath=None) -> None:
        """Initialize a dataset using a disk cache with maximal size."""
        if size_limit:
            self.size_limit = size_limit
        elif fit_size_limit_filepath:
            file_size = os.path.getsize(fit_size_limit_filepath)
            if file_size > self.size_limit:
                self.size_limit = file_size
                message = (
                    'Temporary directory for caching can be up to '
                    f'{self.size_limit} bytes ({sizeof_fmt(size_limit)}) '
                    'large to fit data.'
                )
                # ResourceWarning is usually filtered by default
                warnings.warn(message, ResourceWarning)
        self.cache_filepath = tempfile.mkdtemp()
        self.cache = dc.Cache(self.cache_filepath, size_limit=self.size_limit)

    def __del__(self):
        """Delete the _CacheDatasource."""
        shutil.rmtree(self.cache_filepath)
