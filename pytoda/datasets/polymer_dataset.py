"""PolymerTokenizerDataset module."""
import pandas as pd
import torch
from numpy import iterable
from torch.utils.data import Dataset

from pytoda.warnings import device_warning

from ..smiles.polymer_language import PolymerTokenizer
from ..types import List, Sequence, Tensor, Tuple, Union
from .smiles_dataset import SMILESDataset


class PolymerTokenizerDataset(Dataset):
    """
    Dataset of SMILES from multiple entities encoded as token indexes.

    Creates a tuple of SMILES datasets, one per given entity (i.e. molecule
    class, e.g monomer and catalysts).
    Rows in the annotation df needs to have column names identical to entities,
    mapping to SMILES in the datasets.

    Uses a PolymerTokenizer
    """

    def __init__(
        self,
        *smi_filepaths: str,
        entity_names: Sequence[str],
        annotations_filepath: str,
        annotations_column_names: Union[List[int], List[str]] = None,
        smiles_language: PolymerTokenizer = None,
        canonical: Union[Sequence[bool], bool] = False,
        augment: Union[Sequence[bool], bool] = False,
        kekulize: Union[Sequence[bool], bool] = False,
        all_bonds_explicit: Union[Sequence[bool], bool] = False,
        all_hs_explicit: Union[Sequence[bool], bool] = False,
        randomize: Union[Sequence[bool], bool] = False,
        remove_bonddir: Union[Sequence[bool], bool] = False,
        remove_chirality: Union[Sequence[bool], bool] = False,
        selfies: Union[Sequence[bool], bool] = False,
        sanitize: Union[Sequence[bool], bool] = True,
        padding: Union[Sequence[bool], bool] = True,
        padding_length: Union[Sequence[int], int] = None,
        iterate_dataset: bool = True,
        backend: str = 'eager',
        device: torch.device = None,
        **kwargs,
    ) -> None:
        """
        Initialize a Polymer dataset.

        All SMILES dataset parameter can be controlled either separately for
        each dataset (by iterable of correct length) or globally (bool/int).

        Args:
            smi_filepaths (Files): paths to .smi files, one per entity
            entity_names (Sequence[str]): List of chemical entities.
            annotations_filepath (str): Path to .csv with the IDs of the
                chemical entities and their properties. Needs to have one
                column per entity name.
            annotations_column_names (Union[List[int], List[str]]): indexes
                (positional or strings) for the annotations. Defaults to None,
                a.k.a. all the columns, except the entity_names are annotation
                labels.
            smiles_language (PolymerTokenizer): a polymer language.
                Defaults to None, in which case a new object is created.
            padding (Union[Sequence[bool], bool]): pad sequences to longest in
                the smiles language. Defaults to True. Controlled either for
                each dataset separately (by iterable) or globally (bool).
            padding_length (Union[Sequence[int], int]): manually sets number of
                applied paddings, applies only if padding is True. Defaults to
                None. Controlled either for each dataset separately (by
                iterable) or globally (int).
            canonical (Union[Sequence[bool], bool]): performs canonicalization
                of SMILES (one original string for one molecule), if True, then
                other transformations (augment etc, see below) do not apply.
            augment (Union[Sequence[bool], bool]): perform SMILES augmentation.
                Defaults to False.
            kekulize (Union[Sequence[bool], bool]): kekulizes SMILES
                (implicit aromaticity only).
                Defaults to False.
            all_bonds_explicit (Union[Sequence[bool], bool]): Makes all bonds
                explicit. Defaults to False, only applies if kekulize = True.
            all_hs_explicit (Union[Sequence[bool], bool]): Makes all hydrogens
                explicit. Defaults to False, only applies if kekulize = True.
            randomize (Union[Sequence[bool], bool]): perform a true
                randomization of SMILES tokens. Defaults to False.
            remove_bonddir (Union[Sequence[bool], bool]): Remove directional
                info of bonds. Defaults to False.
            remove_chirality (Union[Sequence[bool], bool]): Remove chirality
                information. Defaults to False.
            selfies (Union[Sequence[bool], bool]): Whether selfies is used
                instead of smiles, defaults to False.
            sanitize (Union[Sequence[bool], bool]): sanitize (bool): RDKit
                sanitization of the molecule. Defaults to True.
            iterate_dataset (bool): whether to go through all SMILES in the
                dataset to build/extend vocab, find longest sequence, and
                checks the passed padding length if applicable. Defaults to
                True.
            backend (str): memory management backend.
                Defaults to eager, prefer speed over memory consumption.
            device (torch.device): DEPRECATED
            kwargs (dict): additional arguments for dataset constructor.

        NOTE: If a parameter that can be given as Union[Sequence[bool], bool]
        is given as Sequence[bool] of wrong length (!= len(entity_names)), the
        first list item is used for all datasets.
        """
        device_warning(device)
        self.backend = backend

        if len(entity_names) != len(smi_filepaths):
            raise ValueError('Give 1 .smi file per entity')

        # Setup parameter

        (
            self.paddings,
            self.padding_lengths,
            self.canonicals,
            self.augments,
            self.kekulizes,
            self.all_bonds_explicits,
            self.all_hs_explicits,
            self.randomizes,
            self.remove_bonddirs,
            self.remove_chiralitys,
            self.selfies,
            self.sanitize,
        ) = map(
            (
                lambda x: x
                if iterable(x) and len(x) == len(entity_names)
                else [x] * len(entity_names)
            ),
            (
                padding,
                padding_length,
                canonical,
                augment,
                kekulize,
                all_bonds_explicit,
                all_hs_explicit,
                randomize,
                remove_bonddir,
                remove_chirality,
                selfies,
                sanitize,
            ),
        )

        if smiles_language is None:
            self.smiles_language = PolymerTokenizer(  # defaults to add smiles
                entity_names=entity_names,
                padding=self.paddings[0],
                padding_length=self.padding_lengths[0],
                canonical=self.canonicals[0],
                augment=self.augments[0],
                kekulize=self.kekulizes[0],
                all_bonds_explicit=self.all_bonds_explicits[0],
                all_hs_explicit=self.all_hs_explicits[0],
                randomize=self.randomizes[0],
                remove_bonddir=self.remove_bonddirs[0],
                remove_chirality=self.remove_chiralitys[0],
                selfies=self.selfies[0],
                sanitize=self.sanitize[0],
                add_start_and_stop=True,
            )
            for index, entity in enumerate(entity_names):
                self.smiles_language.set_smiles_transforms(
                    entity,
                    canonical=self.canonicals[index],
                    augment=self.augments[index],
                    kekulize=self.kekulizes[index],
                    all_bonds_explicit=self.all_bonds_explicits[index],
                    all_hs_explicit=self.all_hs_explicits[index],
                    remove_bonddir=self.remove_bonddirs[index],
                    remove_chirality=self.remove_chiralitys[index],
                    selfies=self.selfies[index],
                    sanitize=self.sanitize[index],
                )
                # set_encoding_transforms only after adding smiles
                # while transforms are needed to add_dataset

        else:
            self.smiles_language = smiles_language

        self.entities = self.smiles_language.entities
        self.datasets = [
            SMILESDataset(smi_filepath, name=self.entities[index], **kwargs)
            for index, smi_filepath in enumerate(smi_filepaths)
        ]

        if iterate_dataset:
            for dataset in self.datasets:
                self.smiles_language.update_entity(dataset.name)
                self.smiles_language.add_dataset(dataset)
            if padding and None in self.padding_lengths:
                # take care, this will call a transform reset
                self.smiles_language.set_max_padding()
        self.smiles_language.current_entity = None

        if smiles_language is None:
            for index, entity in enumerate(entity_names):
                # smiles_transforms might have been reset
                self.smiles_language.set_smiles_transforms(
                    entity,
                    canonical=self.canonicals[index],
                    augment=self.augments[index],
                    kekulize=self.kekulizes[index],
                    all_bonds_explicit=self.all_bonds_explicits[index],
                    all_hs_explicit=self.all_hs_explicits[index],
                    remove_bonddir=self.remove_bonddirs[index],
                    remove_chirality=self.remove_chiralitys[index],
                    selfies=self.selfies[index],
                    sanitize=self.sanitize[index],
                )
                self.smiles_language.set_encoding_transforms(
                    entity,
                    randomize=self.randomizes[index],
                    add_start_and_stop=True,
                    padding=self.paddings[index],
                    padding_length=self.padding_lengths[index],
                )

        # Read and post-process the annotations dataframe
        self.annotations_filepath = annotations_filepath
        self.annotated_data_df = pd.read_csv(self.annotations_filepath)
        # Cast the column names to uppercase
        self.annotated_data_df.columns = map(
            lambda x: str(x).capitalize(), self.annotated_data_df.columns
        )
        columns = self.annotated_data_df.columns

        # handle annotation index
        assert all(
            [entity in columns for entity in self.entities]
        ), 'Some of the chemical entities were not found in the label csv.'

        # handle labels
        if annotations_column_names is None:
            self.labels = [column for column in columns if column not in self.entities]
        elif all([isinstance(column, int) for column in annotations_column_names]):
            self.labels = columns[annotations_column_names]
        elif all([isinstance(column, str) for column in annotations_column_names]):
            self.labels = list(map(lambda x: x.capitalize(), annotations_column_names))
        else:
            raise RuntimeError(
                'label_columns should be an iterable containing int or str'
            )
        # get the number of labels
        self.number_of_tasks = len(self.labels)

        # filter data based on the availability
        masks = []
        mask = pd.Series(
            [True] * len(self.annotated_data_df), index=self.annotated_data_df.index
        )
        for entity, dataset in zip(self.entities, self.datasets):
            # prune rows (in mask) with ids unavailable in respective dataset
            local_mask = self.annotated_data_df[entity].isin(set(dataset.keys()))
            mask = mask & local_mask
            masks.append(local_mask)

        self.annotated_data_df = self.annotated_data_df.loc[mask]

        # to investigate missing ids per entity
        self.masks_df = pd.concat(masks, axis=1)
        self.masks_df.columns = self.entities

    def __len__(self) -> int:
        """Total number of samples."""
        return len(self.annotated_data_df)

    def __getitem__(self, index: int) -> Tuple[Tensor, ...]:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            Tuple: a tuple containing self.entities+1 torch.Tensors
            representing respectively: compound token indexes for each chemical
            entity and the property labels (annotations)
        """

        # sample selection for all entities/datasets
        selected_sample = self.annotated_data_df.iloc[index]

        # labels (annotations)
        labels_tensor = torch.tensor(
            list(selected_sample[self.labels].values),
            dtype=torch.float,
        )
        # samples (SMILES token indexes)
        smiles_tensors = tuple(
            self.smiles_language.smiles_to_token_indexes(
                dataset.get_item_from_key(selected_sample[dataset.name]), dataset.name
            )
            for dataset in self.datasets
        )
        return tuple([*smiles_tensors, labels_tensor])
