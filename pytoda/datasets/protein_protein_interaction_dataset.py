"""Implementation of ProteinProteinInteractionDataset."""
from typing import Iterable, List, Union

import pandas as pd
import torch
from numpy import iterable
from torch.utils.data import Dataset

from ..proteins.protein_language import ProteinLanguage
from ..types import DrugSensitivityData, GeneList
from .annotated_dataset import AnnotatedDataset
from .protein_sequence_dataset import ProteinSequenceDataset


class ProteinProteinInteractionDataset(Dataset):
    """
    PPI Dataset implementation. Designed for two sources of protein sequences
    and on source of discrete labels.
    NOTE: Only supports classification (possibly multitask) but no regression
    tasks.
    """

    def __init__(
        self,
        sequence_filepaths: Union[Iterable[str], Iterable[Iterable[str]]],
        entity_names: Iterable[str],
        labels_filepath: str,
        annotations_column_names: Union[List[int], List[str]] = None,
        protein_language: ProteinLanguage = None,
        amino_acid_dict: str = 'iupac',
        paddings: Union[bool, Iterable[bool]] = True,
        padding_lengths: Union[int, Iterable[int]] = None,
        add_start_and_stops: Union[bool, Iterable[bool]] = False,
        augment_by_reverts: Union[bool, Iterable[bool]] = False,
        randomizes: Union[bool, Iterable[bool]] = False,
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
    ) -> None:
        """
        Initialize a protein protein interactiondataset.

        Args:
            sequence_filepaths (Union[Iterable[str], Iterable[Iterable[str]]]):
                paths to .smi or .csv file for protein sequences. For each item
                in the iterable, one protein sequence dataset is created.
                Iterables can be nested, i.e. each protein sequence dataset can
                be created from an iterable of filepaths.
            entity_names (Iterable[str]): List of protein sequence entities,
                e.g. ['Peptides', 'T-Cell-Receptors']. These names should be
                column names of the labels_filepaths.
            labels_filepath (str): path to .csv file with classification
                labels.
            annotations_column_names (Union[List[int], List[str]]): indexes
                (positional or strings) for the annotations. Defaults to None,
                a.k.a. all the columns, except the entity_names are annotation
                labels.
            protein_language (ProteinLanguage): a protein language, defaults to
                None.
            amino_acid_dict (str): The type of amino acid dictionary to map
                sequence tokens to numericals. Defaults to 'iupac', alternative
                is 'unirep'.
            paddings (Union[bool, Iterable[bool]]): pad sequences to longest in
                the protein language. Defaults to True.
            padding_lengths (Union[int, Iterable[int]]): manually sets number
                of applied paddings (only if padding = True). Defaults to None.
            add_start_and_stops (Union[bool, Iterable[bool]]): add start and
                stop token indexes.  Defaults to False.
            augment_by_reverts (Union[bool, Iterable[bool]]): perform a
                stochastic reversion of the amino acid sequence.
            randomizes (Union[bool, Iterable[bool]]): perform a true
                randomization of the amino acid sequences. Defaults to False.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
        """
        Dataset.__init__(self)
        assert (
            len(entity_names) == len(sequence_filepaths)
        ), 'sequence_filepaths should be an iterable of length in entity names'

        self.sequence_filepaths = sequence_filepaths
        self.labels_filepath = labels_filepath
        self.entities = list(map(lambda x: x.capitalize(), entity_names))

        # device
        self.device = device

        (
            self.paddings, self.padding_lengths, self.add_start_and_stop,
            self.augment_by_reverts, self.randomizes
        ) = map(
            (lambda x: x if iterable(x) and len(x) == 2 else [x] * 2),
            (paddings, padding_lengths, augment_by_reverts, randomizes)
        )

        if protein_language is None:
            self.protein_language = ProteinLanguage()

        else:
            self.protein_language = protein_language
            assert (
                (
                    self.protein_language.add_start_and_stop ==
                    all(add_start_and_stops)
                ) and all(add_start_and_stops) == any(add_start_and_stops)
            ), 'Inconsistencies found in add_start_and_stop.'

        # Create protein sequence datasets
        self._datasets = [
            ProteinSequenceDataset(
                self.sequence_filepaths[index],
                protein_language=protein_language,
                padding=self.paddings[index],
                padding_length=self.padding_lengths[index],
                add_start_and_stop=self.add_start_and_stop[index],
                augment=self.augment_by_reverts[index],
                randomize=self.randomizes[index],
                device=self.device,
                name=self.entities[index]
            ) for index in range(len(self.sequence_filepaths))
        ]
        # Labels
        self.labels_df = pd.read_csv(self.labels_filepath)
        # Cast the column names to uppercase
        self.annotated_data_df.columns = map(
            lambda x: str(x).capitalize(), self.labels_df.columns
        )
        columns = self.annotated_data_df.columns

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
        self.available_sequence_ids = [
            set(dataset.sample_to_index_mapping.keys())
            & set(self.labels_df[entity])
            for entity, dataset in zip(self.entities, self._datasets)
        ]

        available_sequence_ids = []
        for entity, dataset in zip(self.entities, self._datasets):
            available_sequence_ids.append(
                set(dataset.sample_to_index_mapping.keys())
                & set(self.labels_df[entity])
            )

            self.labels_df = self.labels_df.loc[self.labels_df[entity].isin(
                available_sequence_ids[-1]
            )]

        self.available_sequence_ids = available_sequence_ids
        self.number_of_samples = self.labels_df.shape[0]

    def __len__(self) -> int:
        "Total number of samples."
        return self.number_of_samples


def __getitem__(self, index: int) -> Iterable[torch.tensor]:
    """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            Tuple: a tuple containing self.entities+1 torch.Tensors
            representing respetively: compound token indexes for each protein
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
    # samples (Protein sequences)
    proteins_tensors = tuple(
        map(
            lambda x: x[
                x.sample_to_index_mapping[selected_sample[x.name]]
            ],
            self._datasets
        )
    )  # yapf: disable
    return tuple([*proteins_tensors, labels_tensor])
