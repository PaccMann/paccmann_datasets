"""Implementation of ProteinProteinInteractionDataset."""
import pandas as pd
import torch
from numpy import iterable
from torch.utils.data import Dataset

from pytoda.warnings import device_warning

from ..proteins.protein_language import ProteinLanguage
from ..types import Files, List, Sequence, Tensor, Tuple, Union
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
        sequence_filepaths: Union[Files, Sequence[Files]],
        entity_names: Sequence[str],
        labels_filepath: str,
        sequence_filetypes: Union[str, List[str]] = 'infer',
        annotations_column_names: Union[List[int], List[str]] = None,
        protein_languages: Union[ProteinLanguage, List[ProteinLanguage]] = None,
        paddings: Union[bool, Sequence[bool]] = True,
        padding_lengths: Union[int, Sequence[int]] = None,
        add_start_and_stops: Union[bool, Sequence[bool]] = False,
        augment_by_reverts: Union[bool, Sequence[bool]] = False,
        randomizes: Union[bool, Sequence[bool]] = False,
        iterate_datasets: Union[bool, Sequence[bool]] = False,
        device: torch.device = None,
    ) -> None:
        """
        Initialize a protein protein interactiondataset.

        Args:
            sequence_filepaths (Union[Files, Sequence[Files]]):
                paths to .smi (also as .csv) or .fasta (.gz) file for protein
                sequences. For each item in the iterable, one protein sequence
                dataset is created. Sequences can be nested, i.e. each protein
                sequence dataset can be created from an iterable of filepaths
                of same type, see sequence_filetypes.
            entity_names (Sequence[str]): List of protein sequence entities,
                e.g. ['Peptides', 'T-Cell-Receptors']. These names should be
                column names of the labels_filepaths in order respective to
                sequence_filepaths.
            labels_filepath (str): path to .csv file with classification
                labels.
            sequence_filetypes (Union[str, List[str]]): the filetypes of the
                sequence files. Can either be a str if all files have identical
                types or an Sequence if different entities have different
                types. Different types across the same entity are not
                supported. Supported formats are {.smi, .csv, .fasta,
                .fasta.gz}. Default is `infer`, i.e. filetypes are inferred
                automatically.
            annotations_column_names (Union[List[int], List[str]]): indexes
                (positional or strings) for the annotations. Defaults to None,
                a.k.a. all the columns, except the entity_names are annotation
                labels.
            protein_languages (Union[ProteinLanguage, List[ProteinLanguage]):
                one or multiple ProteinLanguages. If multiple are provided, exactly one
                should be given for each entity. If only one is provided, the same
                language will be used for all entities. You can also pass child instances
                like ProteinFeatureLanguage. Defaults to None, i.e., creating a single
                protein language with iupac dictionary for all entities.
            paddings (Union[bool, Sequence[bool]]): pad sequences to longest in
                the protein language. Defaults to True.
            padding_lengths (Union[int, Sequence[int]]): manually sets number
                of applied paddings (only if padding = True). Defaults to None.
            add_start_and_stops (Union[bool, Sequence[bool]]): add start and
                stop token indexes.  Defaults to False.
            augment_by_reverts (Union[bool, Sequence[bool]]): perform a
                stochastic reversion of the amino acid sequence.
            randomizes (Union[bool, Sequence[bool]]): perform a true
                randomization of the amino acid sequences. Defaults to False.
            iterate_datasets (Union[bool, Sequence[bool]]): whether to go through all
                items in the datasets to detect unknown characters, find longest
                sequence and checks passed padding length if applicable.
                Defaults to False.
            device (torch.device): DEPRECATED

        """
        Dataset.__init__(self)
        device_warning(device)
        assert len(entity_names) == len(
            sequence_filepaths
        ), 'sequence_filepaths should be an iterable of length in entity names'

        self.labels_filepath = labels_filepath
        self.entities = list(map(lambda x: x.capitalize(), entity_names))

        # wrap single filepath per entity to treat equally as iterable (*args)
        self.sequence_filepaths = [
            [filepath] if isinstance(filepath, str) else filepath
            for filepath in sequence_filepaths
        ]
        #  Data type of first sequence files per entity
        if sequence_filetypes == 'infer':
            self.filetypes = list(
                map(lambda x: '.' + x[0].split('.')[-1], self.sequence_filepaths)
            )
        elif sequence_filetypes in ['.smi', '.csv', '.fasta', '.fasta.gz']:
            self.filetypes = [sequence_filetypes] * len(self.entities)
        elif len(sequence_filetypes) == len(self.entities) and all(
            x in ['.smi', '.csv', '.fasta', '.fasta.gz'] for x in sequence_filetypes
        ):
            self.filetypes = sequence_filetypes
        else:
            raise ValueError(f'Unsupported filetype: {sequence_filetypes}')

        (
            self.paddings,
            self.padding_lengths,
            self.add_start_and_stops,
            self.augment_by_reverts,
            self.randomizes,
            self.iterate_datasets,
        ) = map(
            (
                lambda x: x
                if iterable(x) and len(x) == len(self.entities)
                else [x] * len(self.entities)
            ),
            (
                paddings,
                padding_lengths,
                add_start_and_stops,
                augment_by_reverts,
                randomizes,
                iterate_datasets,
            ),
        )

        if protein_languages is None:
            # Objects will be constructed in `ProteinSequenceDataset`
            self.protein_languages = [None for _ in self.entities]
            self.pl_methods = ['iupac' for _ in self.entities]
        else:
            if isinstance(protein_languages, Sequence):
                # Multiple languages were passed
                self.protein_languages = protein_languages
                self.pl_methods = [p.method for p in protein_languages]

            elif isinstance(protein_languages, ProteinLanguage):
                # Single language was passed
                self.protein_languages = [protein_languages for _ in self.entities]
                self.pl_methods = [protein_languages.method for _ in self.entities]

            else:
                raise TypeError(
                    f'Received unknown type for protein_language: {type(protein_languages)}'
                )
            # Check whether input arguments are consistent
            for i, (lang, start) in enumerate(
                zip(self.protein_languages, self.add_start_and_stops)
            ):
                assert (
                    lang.add_start_and_stop == start
                ), f'add_start_and_stop differs for language {i}: {start} vs. {lang.add_start_and_stop}'

        # Create protein sequence datasets.
        self.datasets = [
            ProteinSequenceDataset(
                *filepaths,
                filetype=self.filetypes[index],
                protein_language=self.protein_languages[index],
                padding=self.paddings[index],
                padding_length=self.padding_lengths[index],
                add_start_and_stop=self.add_start_and_stops[index],
                augment_by_revert=self.augment_by_reverts[index],
                randomize=self.randomizes[index],
                name=self.entities[index],
                iterate_dataset=self.iterate_datasets[index],
            )
            for index, filepaths in enumerate(self.sequence_filepaths)
        ]
        # Retrieve the possibly updated protein languages
        self.protein_languages = [data.protein_language for data in self.datasets]

        # Labels
        self.labels_df = pd.read_csv(self.labels_filepath)
        # Cast the column names to uppercase
        self.labels_df.columns = map(
            lambda x: str(x).capitalize(), self.labels_df.columns
        )
        columns = self.labels_df.columns

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

        assert all(
            list(map(lambda x: x in self.labels_df.columns, self.entities))
        ), 'At least one given entity name was not found in labels_filepath.'

        # filter data based on the availability
        masks = []
        mask = pd.Series([True] * len(self.labels_df), index=self.labels_df.index)

        for entity, dataset in zip(self.entities, self.datasets):
            # prune rows (in mask) with ids unavailable in respective dataset
            local_mask = self.labels_df[entity].isin(set(dataset.keys()))
            mask = mask & local_mask
            masks.append(local_mask)

        self.labels_df = self.labels_df.loc[mask]

        # to investigate missing ids per entity
        self.masks_df = pd.concat(masks, axis=1)
        self.masks_df.columns = self.entities

        self.number_of_samples = len(self.labels_df)

    def __len__(self) -> int:
        "Total number of samples."
        return self.number_of_samples

    def __getitem__(self, index: int) -> Tuple[Tensor, ...]:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            Tuple: a tuple containing self.entities+1 torch.Tensors
            representing respectively: compound token indexes for each
            protein entity and the property labels (annotations)
        """

        # sample selection
        selected_sample = self.labels_df.iloc[index]

        # labels (annotations)
        labels_tensor = torch.tensor(
            list(selected_sample[self.labels].values),
            dtype=torch.float,
        )
        # samples (Protein sequences)
        proteins_tensors = [
            ds.get_item_from_key(selected_sample[ds.name]) for ds in self.datasets
        ]
        return tuple([*proteins_tensors, labels_tensor])
