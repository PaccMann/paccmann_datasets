"""Utilities for the tests."""
import os
import tempfile
import warnings


class TestFileContent:
    """
    Create a temporary file with a given content.

    Inspired by: https://stackoverflow.com/a/54053967/10032558.
    """

    __test__ = False  # avoid PytestCollectionWarning

    def __init__(self, content: str, **kwargs) -> None:
        """
        Initialize the file with a content.

        Args:
            content (str): content of the file.
            **kwargs (dict): Additional keyword arguments for NamedTemporaryFile.
                NOTE: This can e.g. be suffix='.csv' if the temporary filename should
                adhere to a specific suffix.
        """
        self.file = tempfile.NamedTemporaryFile(mode='w', delete=False, **kwargs)
        with self.file as fp:
            fp.write(content)

    @property
    def filename(self) -> str:
        """
        Get the name of the file.

        Returns:
            str: the file name.
        """
        return self.file.name

    def __enter__(self) -> object:
        """Enter the `with` block."""
        return self

    def __exit__(self, type, value, traceback) -> None:
        """Exit the `with` block."""
        try:
            os.remove(self.file.name)
        except Exception:
            warnings.warn(f'File {self.file.name} could not be closed.')
            self.file.close()
