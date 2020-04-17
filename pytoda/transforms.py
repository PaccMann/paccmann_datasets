"""Transform utilities."""
import logging
import random
from copy import deepcopy
from typing import Any

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
            Indexes: left padded indexes representation.
        """
        if self.padding_length < len(token_indexes):
            logger.warning(
                f'\n{token_indexes} is longer than padding length '
                f'({self.padding_length}). End of string will be stripped off.'
            )
            return token_indexes[:self.padding_length]
        else:
            return (
                (self.padding_length - len(token_indexes)) *
                [self.padding_index] + token_indexes
            )


class ToTensor(Transform):
    """Transform token indexes to torch tensor."""

    def __init__(self, device, dtype=torch.short) -> None:
        """
        Initialize a token indexes to tensor object.

        Args:
            dtype (torch.dtype): data type. Defaults to torch.short.
            device (torch.device): device where the tensors are stored.
            Defaults to gpu, if available.
        """
        self.dtype = torch.short
        self.device = device

    def __call__(self, token_indexes: Indexes) -> torch.Tensor:
        """
        Apply the transform.

        Args:
            token_indexes (Indexes): token indexes.

        Returns:
            torch.Tensor: tensor representation of the token indexes.
        """
        return torch.tensor(
            token_indexes, dtype=self.dtype, device=self.device
        ).view(-1, 1).squeeze()


class Randomize(Transform):
    """Randomize a molecule by truly shuffling all tokens."""

    def __call__(self, tokens: Indexes) -> Indexes:
        """
        Intialize SMILES randomizer.

        NOTE: Must not apply this transformation on SMILES string, only on the
            tokenized, numerical vectors (i.e. after SMILESToTokenIndexes)

        Args:
            tokens (Indexes): indexes representation for the SMILES to be
                randomized.
        Returns:
           Indexes: shuffled indexes representation of the molecule
        """
        smiles_tokens = deepcopy(tokens)
        np.random.shuffle(smiles_tokens)
        return smiles_tokens


class AugmentByReversing(Transform):
    """Augment an amino acid sequence by (eventually) flipping order"""

    def __call__(self, sequence: str) -> str:
        """
        Apply the transform.

        Args:
            sequnce (str): a sequence representation.

        Returns:
            str: Either the sequence itself, or the revesed sequence.
        """
        return sequence[::-1] if round(random.random()) else sequence


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
