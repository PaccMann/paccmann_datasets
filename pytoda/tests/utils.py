"""Utilities for the tests."""
import os
import tempfile


class TestFileContent:
    """
    Create a temporary file with a given content.

    Inspired by: https://stackoverflow.com/a/54053967/10032558.
    """

    def __init__(self, content: str) -> None:
        """
        Initialize the file with a content.

        Args:
            content (str): content of the file.
        """
        self.file = tempfile.NamedTemporaryFile(mode='w', delete=False)
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
        os.unlink(self.filename)
