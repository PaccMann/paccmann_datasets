"""Type definitions."""
import inspect
from typing import (Any, Callable, Hashable, Iterable, Iterator, List,  # noqa
                    Sequence, Tuple, Union)

from torch import Tensor

Tokens = List[str]
Indexes = List[int]
SMILESTokenizer = Callable[[str], Tokens]
FileList = Sequence[str]  # often passed on as Tuple via *args
GeneList = List[str]
FeatureList = List[str]
TransformList = List[Callable[[Any], Any]]
DrugSensitivityData = Tuple[Tensor, Tensor, Tensor]
DrugAffinityData = Tuple[Tensor, Tensor, Tensor]
AnnotatedData = Tuple[Any, Tensor]


def delegate_kwargs(to=None, keep=False):
    """
    Decorator: replace `**kwargs` in signature with params from `to`.

    Source: https://www.fast.ai/2019/08/06/delegation/
    """

    def _f(f):
        if to is None:
            to_f, from_f = f.__base__.__init__, f.__init__
        else:
            to_f, from_f = to, f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop('kwargs')
        s2 = {
            k: v
            for k, v in inspect.signature(to_f).parameters.items()
            if v.default != inspect.Parameter.empty and k not in sigd
        }
        sigd.update(s2)
        if keep:
            sigd['kwargs'] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f

    return _f
