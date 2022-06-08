"""Transform utilities."""
import logging
import random
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import torch

from .types import Indexes, TransformList

logger = logging.getLogger('pytoda_transforms')


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

    def __eq__(self, other: object) -> bool:
        """Equality comparison of Transform objects. Two transform instances are
        identical if the transforms are identical

        Args:
            other (Compose): Compose object for comparison.

        Returns:
            bool: Whether objects are identical
        """
        return vars(self) == vars(other)


class StartStop(Transform):
    """Add start and stop token indexes at beginning and end of sequence."""

    def __init__(self, start_index: int, stop_index: int):
        """
        Initialize a left padding token indexes object.

        Args:
            start_index (int): index of start token in vocabulary.
            start_index (int): index of stop token in vocabulary.
        """
        self.start_index = start_index
        self.stop_index = stop_index

    def __call__(self, token_indexes: Indexes) -> Indexes:
        """
        Apply the transform.

        Args:
            token_indexes (Indexes): token indexes.

        Returns:
            Indexes: Indexes representation with start and stop added.
        """
        return [self.start_index] + token_indexes + [self.stop_index]


class LeftPadding(Transform):
    """Left pad token indexes."""

    def __init__(self, padding_length: int, padding_index: int) -> None:
        """
        Initialize a left padding token indexes object.

        Args:
            padding_length (int): length of the padding.
            padding_index (int): padding index.
        """
        self.padding_length = padding_length
        self.padding_index = padding_index

    def __call__(self, token_indexes: Indexes) -> Indexes:
        """
        Apply the transform.

        Args:
            token_indexes (Indexes): token indexes.

        Returns:
            Indexes: indexes representation with given `padding_length`.
                token_indexes is cut short or left padded with `padding_index`.
        """
        try:
            if self.padding_length < len(token_indexes):
                logger.warning(
                    f'\n{token_indexes} is longer than padding length '
                    f'({self.padding_length}). End of string will be stripped '
                    'off.'
                )
                return token_indexes[: self.padding_length]
            else:
                return (self.padding_length - len(token_indexes)) * [
                    self.padding_index
                ] + token_indexes
        except TypeError as e:
            if self.padding_length is None:
                raise TypeError(
                    'padding_length=None was given but integer is required.'
                )
            else:
                raise e


class ToTensor(Transform):
    """Transform token indexes to torch tensor."""

    def __init__(self, dtype: torch.dtype = torch.short) -> None:
        """
        Initialize a token indexes to tensor object.

        Args:
            dtype (torch.dtype): data type. Defaults to torch.short.
        """

        if not isinstance(dtype, torch.dtype):
            raise TypeError(f'Dtype must be torch.dtype not {type(dtype)}')
        self.dtype = dtype

    def __call__(self, token_indexes: Indexes) -> torch.Tensor:
        """
        Apply the transform. Convert token_indexes (e.g. list) to a torch Tensor.

        Args:
            token_indexes (Indexes): token indexes.

        Returns:
            torch.Tensor: tensor representation of the token indexes.
        """
        return torch.tensor(token_indexes, dtype=self.dtype).view(-1, 1).squeeze()


class ListToTensor(Transform):
    """
    2D Version of ToTensor.
    """

    def __init__(self, dtype: torch.dtype = torch.float) -> None:
        """
        Initialize a token indexes to tensor object.

        Args:

            dtype (torch.dtype): data type. Defaults to torch.float.
        """
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f'Dtype must be torch.dtype not {type(dtype)}')
        self.dtype = dtype

    def __call__(self, token_indexes: Indexes) -> torch.Tensor:
        """
        Apply the transform (Transform a list of token indexes to torch tensor).
            2D version of ToTensor.

        Args:
            token_indexes (Indexes): token indexes.

        Returns:
            torch.Tensor: tensor representation of the token indexes.
        """
        return torch.tensor(token_indexes, dtype=self.dtype)


class Randomize(Transform):
    """Randomize a sequence all tokens."""

    def __call__(self, tokens: Indexes) -> Indexes:
        """
        Args:
            tokens (Indexes): indexes representation for the SMILES to be
                randomized.
        Returns:
           Indexes: shuffled indexes representation of the molecule

        NOTE: If this transform is used on SMILES, it must not be applied on
            the raw SMILES string, but on the tokenized, numerical vectors
            (i.e. after SMILESToTokenIndexes).
        """
        smiles_tokens = deepcopy(tokens)
        np.random.shuffle(smiles_tokens)
        return smiles_tokens


class AugmentByReversing(Transform):
    """Augment an sequence by (eventually) flipping order"""

    def __init__(self, p: float = 0.5) -> None:
        """
        AugmentByReversing constructor.

        Args:
            p (float): Probability that reverting occurs.

        """
        if not isinstance(p, float):
            raise TypeError(f'Please pass float, not {type(p)}.')
        self.p = np.clip(p, 0.0, 1.0)

    def __call__(self, sequence: str) -> str:
        """
        Apply the transform.

        Args:
            sequnce (str): a sequence representation.

        Returns:
            str: Either the sequence itself, or the revesed sequence.
        """
        return sequence[::-1] if random.random() < self.p else sequence


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

    def __eq__(self, other: Transform) -> bool:
        """Equality comparison of Compose objects. Two compose instances are identical
        if the transforms are identical

        Args:
            other (Compose): Compose object for comparison.

        Returns:
            bool: Whether objects are identical
        """
        if len(self.transforms) != len(other.transforms):
            return False
        else:
            return all([a == b for a, b in zip(self.transforms, other.transforms)])


class ExtractFromDict(Transform):
    """Extracts a value from a dictionary based on a key set in the constructor."""

    def __init__(self, key: str) -> None:
        """
        Args:
            key: The key used to access the dictionary
        """
        self.key = key

    def __call__(self, sample_dict: Dict[str, str]) -> str:
        """
        Extract value from a str-str-dictionary.

        Args:
            sample_dict: A dictionary containing the key set in constructor.

        Returns:
            The extracted string.
        """
        assert self.key in sample_dict, f'Key {self.key} not found in sample_dict'
        return sample_dict[self.key]


class DiscardLowercase(Transform):
    """Discard lower-case letters (and non-letter characters) from a sequence."""

    def __init__(
        self,
    ) -> None:
        """ """
        pass

    def __call__(self, sequence: str) -> str:
        """
        Apply the transform.

        Args:
            sequence (str):

        Returns:
            str:
        """
        ans = ''.join([x for x in sequence if (x >= 'A') and (x <= 'Z')])
        return ans


class ToUpperCase(Transform):
    """Convert all characters to uppercase."""

    def __init__(
        self,
    ) -> None:
        pass

    def __call__(self, sequence: str) -> str:
        """
        Apply the transform.

        Args:
            sequence (str): A sequence.

        Returns:
            str: A sequence in uppercase.
        """
        ans = sequence.upper()
        return ans
