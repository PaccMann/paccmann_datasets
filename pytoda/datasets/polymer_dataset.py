"""PolymerDataset module."""
from copy import deepcopy
from typing import Iterable, List, Union

import pandas as pd
import torch
from numpy import iterable
from ..smiles.transforms import SMILESToTokenIndexes
from ..smiles.polymer_language import PolymerLanguage
from .smiles_dataset import SMILESDataset


class PolymerDataset(SMILESDataset):
    """
    Polymer dataset implementation.

    Creates a tuple of SMILES datasets, one per given entity (i.e. molecule
    class, e.g monomer and catalysts).
    The annotation df needs to have column names identical to entities.

    NOTE:
    All SMILES dataset parameter can be controlled either separately for each
    dataset (by iterable of correct length) or globally (bool/int).

    """

    def __init__(
        self,
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
        sanitize: Union[Iterable[str], bool] = True,
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
        backend: str = 'eager'
    ) -> None:
        """
        Initialize a Polymer dataset.

        Args:
            smi_filepaths (FileList): paths to .smi files, one per entity
            entity_names (Iterable[str]): List of chemical entities.
            annotations_filepath (str): Path to .csv with the IDs of the
                chemical entities and their properties. Needs to have one
                column per entity name.
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
            sanitize (Union[Iterable[str], bool]): Sanitize SMILES.
                Defaults to True.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
            backend (str): memory management backend.
                Defaults to eager, prefer speed over memory consumption.
        """

        self.device = device
        self.backend = backend

        assert (
            len(entity_names) == len(smi_filepaths)
        ), 'Give 1 .smi file per entity'

        if smiles_language is None:
            self.smiles_language = PolymerLanguage(entity_names=entity_names)
        else:
            self.smiles_language = smiles_language
        self.entities = self.smiles_language.entities

        # Setup parameter
        # NOTE: If a parameter that can be given as Union[Iterable[str], bool]
        # is given as Iterable[str] of wrong length (!= len(entity_names)), the
        # first list item is used for all datasets.
        (
            self.paddings, self.padding_lengths, self.canonicals,
            self.augments, self.kekulizes, self.all_bonds_explicits,
            self.all_hs_explicits, self.randomizes, self.remove_bonddirs,
            self.remove_chiralitys, self.selfies, self.sanitize
        ) = map(
            (
                lambda x: x if iterable(x) and len(x) == len(self.entities)
                else [x] * len(self.entities)
            ), (
                padding, padding_length, canonical, augment, kekulize,
                all_bonds_explicit, all_hs_explicit, randomize, remove_bonddir,
                remove_chirality, selfies, sanitize
            )
        )
        # Create one SMILES dataset per chemical entity
        self._datasets = [
            SMILESDataset(
                smi_filepaths[index],
                name=self.entities[index],
                smiles_language=self.smiles_language,
                padding=self.paddings[index],
                padding_length=self.padding_lengths[index],
                canonical=self.canonicals[index],
                augment=self.augments[index],
                kekulize=self.kekulizes[index],
                all_bonds_explicit=self.all_bonds_explicits[index],
                all_hs_explicit=self.all_hs_explicits[index],
                randomize=self.randomizes[index],
                remove_bonddir=self.remove_bonddirs[index],
                remove_chirality=self.remove_chiralitys[index],
                selfies=self.selfies[index],
                sanitize=self.sanitize[index],
                device=device
            ) for index in range(len(smi_filepaths))
        ]
        """
        Push the Polymer language configuration down to the smiles language
        object associated to the dataset and to the tokenizer that use this
        object.
        """
        for dataset in self._datasets:

            dataset.smiles_language = deepcopy(self.smiles_language)
            dataset.smiles_language.update_entity(dataset.name)
            tokenizer_index = [
                i for i, t in enumerate(dataset._dataset.transform.transforms)
                if isinstance(t, SMILESToTokenIndexes)
            ]
            if len(tokenizer_index) > 0:
                dataset._dataset.transform.transforms[
                    tokenizer_index[-1]
                ].smiles_language = dataset.smiles_language  # yapf: disable

        # Read and post-process the annotations dataframe
        self.annotations_filepath = annotations_filepath
        self.annotated_data_df = pd.read_csv(self.annotations_filepath)
        # Cast the column names to uppercase
        self.annotated_data_df.columns = map(
            lambda x: str(x).capitalize(), self.annotated_data_df.columns
        )
        columns = self.annotated_data_df.columns

        # handle annotation index
        assert (
            all([entity in columns for entity in self.entities])
        ), 'Some of the chemical entities were not found in the label csv.'

        # handle labels
        if annotations_column_names is None:
            self.labels = [
                column for column in columns if column not in self.entities
            ]
        elif all(
            [isinstance(column, int) for column in annotations_column_names]
        ):
            self.labels = columns[annotations_column_names]
        elif all(
            [isinstance(column, str) for column in annotations_column_names]
        ):
            self.labels = list(
                map(lambda x: x.capitalize(), annotations_column_names)
            )
        else:
            raise RuntimeError(
                'label_columns should be an iterable containing int or str'
            )
        # get the number of labels
        self.number_of_tasks = len(self.labels)

        # NOTE: filter data based on the availability
        available_entity_ids = []
        for entity, dataset in zip(self.entities, self._datasets):
            available_entity_ids.append(
                set(dataset.sample_to_index_mapping.keys())
                & set(self.annotated_data_df[entity])
            )

            self.annotated_data_df = self.annotated_data_df.loc[
                self.annotated_data_df[entity].isin(available_entity_ids[-1])]

        self.available_entity_ids = available_entity_ids
        self.number_of_samples = self.annotated_data_df.shape[0]

    def __len__(self) -> Iterable[int]:
        """Total number of samples."""
        return self.number_of_samples

    def __getitem__(self, index: int) -> Iterable[torch.tensor]:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            Tuple: a tuple containing self.entities+1 torch.Tensors
            representing respetively: compound token indexes for each chemical
            entity and the property labels (annotations)
        """

        # sample selection
        selected_sample = self.annotated_data_df.iloc[index]

        # labels (annotations)
        labels_tensor = torch.tensor(
            list(selected_sample[self.labels].values),
            dtype=torch.float,
            device=self.device
        )
        # samples (SMILES)
        smiles_tensor = tuple(
            map(
                lambda x: x[
                    x.sample_to_index_mapping[selected_sample[x.name]]
                ],
                self._datasets
            )
        )  # yapf: disable
        return tuple([*smiles_tensor, labels_tensor])
