import numpy as np
import torch

from pytoda.types import Any, Tensor, Tuple


def range_tensor(value_range: Any, repetitions: Tuple[int]) -> Tensor:
    """Returns a background tensor filled with a given range of values.

    Args:
        value_range (Any): Range of values to insert into each row of the
            background tensor.
        repetitions (Tuple[int]): The number of repetitions of value_range along each axis.

    Returns:
        Tensor: Tensor containing repetitions of the given range of values along specified axes.
            Example, value_range = [0,1,2], repetitions = (1,2) will repeat [0,1,2]
            once along dim 0 and twice along dim 1, i.e, tensor([[0,1,2,0,1,2]])
            of size (1,6) will be the output.
        NOTE: if a pattern [0,1,2] is required to fill a tensor of shape (2,5)
              specify value_range as [0,1,2,0,1] and repetiitons as (2,). The
              value_range is filled 'row-wise'. Simply transpose the output for
              a 'column-wise' fill.
    """
    return torch.from_numpy(np.tile(value_range, repetitions))


def constant_value_tensor(value: float, shape: Tuple) -> Tensor:
    """Returns a background tensor filled with a constant value.

    Args:
        value (float): Value to fill the background tensor with.
        shape (Tuple): Shape of the background tensor.

    Returns:
        Tensor: Tensor of given shape filled with the given constant value.
    """
    return torch.full(shape, value)
