"""Transform utilities."""
from typing import Any
from .types import TransformList


class Transform(object):
    """Transform abstract class."""
    def __call__(self, sample: Any) -> Any:
        """
        Apply the transformation.

        Args:
            sample (Any): a sample to transform.

        Returns:
            Any: the transformed sample.
        """
        raise NotImplementedError


class Compose(Transform):
    """
    Composes several transforms together.

    From:
    https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Compose.
    """

    def __init__(self, transforms: TransformList) -> None:
        """
        Initialize a compose transform object.

        Args:
            transforms (TransformList): a list of transforms.
        """
        self.transforms = transforms

    def __call__(self, sample: Any) -> Any:
        """
        Apply the composition of transforms.

        Args:
            sample (Any): a sample to transform.

        Returns:
            Any: the transformed sample.
        """
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __repr__(self) -> str:
        """
        Represent the composition as a string.

        Returns:
            str: a string representing the composed transformation.
        """
        format_string = self.__class__.__name__ + '('
        for transform in self.transforms:
            format_string += '\n'
            format_string += '\t{}'.format(transform)
        format_string += '\n)'
        return format_string
