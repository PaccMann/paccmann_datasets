"""PolymerDataset module."""
from copy import deepcopy
from typing import Iterable, List, Union

import pandas as pd
import torch
from numpy import iterable
from ..smiles.transforms import SMILESToTokenIndexes
from ..smiles.polymer_language import PolymerLanguage
from .smiles_dataset import SMILESDataset

from ._polymer_dataset import \
    _PolymerDatasetAnnotation, _PolymerDatasetNoAnnotation

POLYMER_DATASET_TYPES = {
    'annotated': _PolymerDatasetAnnotation,
    'no_annotated': _PolymerDatasetNoAnnotation
}


def PolymerDataset(
    smi_filepaths: Iterable[str],
    entity_names: Iterable[str],
    annotations_filepath: str,
    annotations_column_names: Union[List[int], List[str]] = None,
    smiles_language: PolymerLanguage = None,
    padding: Union[Iterable[str], bool] = True,
    padding_length: Union[Iterable[str], int] = None,
    canonical: Union[Iterable[str], bool] = False,
    augment: Union[Iterable[str], bool] = False,
    kekulize: Union[Iterable[str], bool] = False,
    all_bonds_explicit: Union[Iterable[str], bool] = False,
    all_hs_explicit: Union[Iterable[str], bool] = False,
    randomize: Union[Iterable[str], bool] = False,
    remove_bonddir: Union[Iterable[str], bool] = False,
    remove_chirality: Union[Iterable[str], bool] = False,
    selfies: Union[Iterable[str], bool] = False,
    device: torch.
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
    backend: str = 'eager'
) -> None:
    """   Polymer dataset implementation.

    Creates a tuple of SMILES datasets, one per given entity (i.e. molecule
    class, e.g monomer and catalysts).
    The annotation df needs to have column names identical to entities.

    NOTE:
    All SMILES dataset parameter can be controlled either separately for each
    dataset (by iterable of correct length) or globally (bool/int).


    Args:
        smi_filepaths (FileList): paths to .smi files, one per entity
        entity_names (Iterable[str]): List of chemical entities.
        annotations_filepath (Union[str, None]): Path to .csv with the IDs of 
            the chemical entities and their properties. Needs to have one
            column per entity name. If `None` is explicitly passed a polymer
            dataset without annotations will be used, that means that the items
            will need to be retrieved by explicitly selecting entity index.
        annotations_column_names (Union[List[int], List[str]]): indexes
            (positional or strings) for the annotations. Defaults to None,
            a.k.a. all the columns, except the entity_names are annotation
            labels.
        smiles_language (PolymerLanguage): a polymer language.
            Defaults to None, in which case a new object is created.
        padding (Union[Iterable[str], bool]): pad sequences to longest in
            the smiles language. Defaults to True. Controlled either for
            each dataset separately (by iterable) or globally (bool).
        padding_length (Union[Iterable[str], int]): manually sets number of
            applied paddings, applies only if padding is True. Defaults to
            None. Controlled either for each dataset separately (by
            iterable) or globally (int).
        canonical (Union[Iterable[str], bool]): performs canonicalization
            of SMILES (one original string for one molecule), if True, then
            other transformations (augment etc, see below) do not apply.
        augment (Union[Iterable[str], bool]): perform SMILES augmentation.
            Defaults to False.
        kekulize (Union[Iterable[str], bool]): kekulizes SMILES
            (implicit aromaticity only).
            Defaults to False.
        all_bonds_explicit (Union[Iterable[str], bool]): Makes all bonds
            explicit. Defaults to False, only applies if kekulize = True.
        all_hs_explicit (Union[Iterable[str], bool]): Makes all hydrogens
            explicit. Defaults to False, only applies if kekulize = True.
        randomize (Union[Iterable[str], bool]): perform a true
            randomization of SMILES tokens. Defaults to False.
        remove_bonddir (Union[Iterable[str], bool]): Remove directional
            info of bonds. Defaults to False.
        remove_chirality (Union[Iterable[str], bool]): Remove chirality
            information. Defaults to False.
        selfies (Union[Iterable[str], bool]): Whether selfies is used
            instead of smiles, defaults to False.
        device (torch.device): device where the tensors are stored.
            Defaults to gpu, if available.
        backend (str): memory management backend.
            Defaults to eager, prefer speed over memory consumption.
    """

    annotated = 'no_annotated'
    if annotations_filepath is not None:
        annotated = 'annotated'

    dataset = POLYMER_DATASET_TYPES[annotated](
        smi_filepaths=smi_filepaths,
        entity_names=entity_names,
        annotations_filepath=annotations_filepath,
        annotations_column_names=annotations_column_names,
        smiles_language=smiles_language,
        padding=padding,
        padding_length=padding_length,
        canonical=canonical,
        augment=augment,
        kekulize=kekulize,
        all_bonds_explicit=all_bonds_explicit,
        all_hs_explicit=all_hs_explicit,
        randomize=randomize,
        remove_bonddir=remove_bonddir,
        remove_chirality=remove_chirality,
        selfies=selfies,
        device=device
    )

    return dataset
