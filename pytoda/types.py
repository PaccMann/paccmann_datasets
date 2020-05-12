"""Type definitions."""
import torch
from typing import List, Callable, Any, Tuple, Union, Hashable, Iterator  # noqa

Tokens = List[str]
Indexes = List[int]
SMILESTokenizer = Callable[[str], Tokens]
FileList = List[str]
GeneList = List[str]
FeatureList = List[str]
TransformList = List[Callable[[Any], Any]]
DrugSensitivityData = Tuple[torch.tensor, torch.tensor, torch.tensor]
DrugAffinityData = Tuple[torch.tensor, torch.tensor, torch.tensor]
AnnotatedData = Tuple[torch.tensor, torch.tensor]
