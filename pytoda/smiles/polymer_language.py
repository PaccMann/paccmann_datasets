"""Polymer language handling."""
from typing import Sequence

from ..types import Indexes, Tensor, Union  # , delegate_kwargs
from .smiles_language import SMILESTokenizer
from .transforms import compose_encoding_transforms, compose_smiles_transforms


# @delegate_kwargs
class PolymerTokenizer(SMILESTokenizer):
    """
    PolymerTokenizer class.

    PolymerTokenizer is an extension of SMILESTokenizer adding special start and
    stop tokens per entity.
    A polymer language is usually shared across several SMILES datasets (e.g.
    different entity sources).
    """

    def __init__(
        self,
        entity_names: Sequence[str],
        name: str = 'polymer-language',
        add_start_and_stop: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize Polymer language able to encode different entities.

        Args:
            entity_names (Sequence[str]): A list of entity names that the
                polymer language can distinguish.
            name (str): name of the PolymerTokenizer.
            add_start_and_stop (bool): add start and stop token indexes.
                Defaults to True.
            kwargs (dict): additional parameters passed to SMILESTokenizer.

        NOTE:
            See `set_smiles_transforms` and `set_encoding_transforms` to change
            the transforms temporarily and reset with
            `reset_initial_transforms`. Assignment of class attributes
            in the parameter list will trigger such a reset.
        """

        super().__init__(name=name, add_start_and_stop=add_start_and_stop, **kwargs)
        self.entities = list(map(lambda x: x.capitalize(), entity_names))
        self.init_kwargs['entity_names'] = self.entities
        self.current_entity = None

        # rebuild basic vocab to group special tokens
        self.start_entity_tokens, self.stop_entity_tokens = (
            list(map(lambda x: '<' + x.upper() + '_' + s + '>', entity_names))
            for s in ['START', 'STOP']
        )

        # required for `token_indexes_to_smiles`
        self.special_indexes.update(
            enumerate(
                self.start_entity_tokens + self.stop_entity_tokens,
                start=len(self.special_indexes),
            )
        )
        self.setup_vocab()

        if kwargs.get('vocab_file', None):
            self.load_vocabulary(kwargs['vocab_file'])

        self.reset_initial_transforms()

    def _check_entity(self, entity: str) -> str:
        entity_ = entity.capitalize()
        if entity_ not in self.entities:
            raise ValueError(f'Unknown entity was given ({entity_})')
        return entity_

    def update_entity(self, entity: str) -> None:
        """
        Update the current entity and the default transforms (used e.g. in
        `add_dataset`) of the Polymer language object.

        Args:
            entity (str): a chemical entity (e.g. 'Monomer').
        """

        self.current_entity = self._check_entity(entity)
        self.transform_smiles = self.all_smiles_transforms[self.current_entity]
        self.transform_encoding = self.all_encoding_transforms[self.current_entity]

    def smiles_to_token_indexes(
        self, smiles: str, entity: str = None
    ) -> Union[Indexes, Tensor]:
        """
        Transform character-level SMILES into a sequence of token indexes.

        In case of add_start_stop, inserts entity specific tokens.

        Args:
            smiles (str): a SMILES (or SELFIES) representation.
            entity (str): a chemical entity (e.g. 'Monomer'). Defaults to
                None, where the current entity is used (initially the
                SMILESTokenizer default).

        Returns:
            Union[Indexes, Tensor]: indexes representation for the
                SMILES/SELFIES provided.
        """
        if entity is None:
            # default behavior given by call to update_entity()
            entity = self.current_entity
        else:
            entity = self._check_entity(entity)

        return self.all_encoding_transforms[entity](
            [
                self.token_to_index.get(token, self.unknown_token)
                for token in self.smiles_tokenizer(
                    self.all_smiles_transforms[entity](smiles)
                )
            ]
        )

    def reset_initial_transforms(self):
        """
        Reset smiles and token indexes transforms as on initialization,
        including entity specific transforms.
        """
        super().reset_initial_transforms()
        if not hasattr(self, 'entities'):  # call from base
            return
        self.current_entity = None
        self.all_smiles_transforms = {
            None: self.transform_smiles,
        }
        self.all_encoding_transforms = {
            None: self.transform_encoding,
        }
        for entity in self.entities:
            self.set_smiles_transforms(entity)
            self.set_encoding_transforms(entity)

    def set_smiles_transforms(
        self,
        entity,
        canonical=None,
        augment=None,
        kekulize=None,
        all_bonds_explicit=None,
        all_hs_explicit=None,
        remove_bonddir=None,
        remove_chirality=None,
        selfies=None,
        sanitize=None,
    ):
        """
        Helper function to reversibly change the transforms per entity.
        """
        entity = self._check_entity(entity)
        self.all_smiles_transforms[entity] = compose_smiles_transforms(
            canonical=canonical if canonical is not None else self.canonical,
            augment=augment if augment is not None else self.augment,
            kekulize=kekulize if kekulize is not None else self.kekulize,
            all_bonds_explicit=all_bonds_explicit
            if all_bonds_explicit is not None
            else self.all_bonds_explicit,
            all_hs_explicit=all_hs_explicit
            if all_hs_explicit is not None
            else self.all_hs_explicit,
            remove_bonddir=remove_bonddir
            if remove_bonddir is not None
            else self.remove_bonddir,
            remove_chirality=remove_chirality
            if remove_chirality is not None
            else self.remove_chirality,
            selfies=selfies if selfies is not None else self.selfies,
            sanitize=sanitize if sanitize is not None else self.sanitize,
        )

    def set_encoding_transforms(
        self,
        entity,
        randomize=None,
        add_start_and_stop=None,
        padding=None,
        padding_length=None,
    ):
        """
        Helper function to reversibly change the transforms per entity.
        Addresses entity specific start and stop tokens.
        """
        entity = self._check_entity(entity)
        start_index = self.token_to_index['<' + entity.upper() + '_START>']
        stop_index = self.token_to_index['<' + entity.upper() + '_STOP>']

        self.all_encoding_transforms[entity] = compose_encoding_transforms(
            randomize=randomize if randomize is not None else self.randomize,
            add_start_and_stop=add_start_and_stop
            if add_start_and_stop is not None
            else self.add_start_and_stop,
            start_index=start_index,
            stop_index=stop_index,
            padding=padding if padding is not None else self.padding,
            padding_length=padding_length
            if padding_length is not None
            else self.padding_length,
            padding_index=self.padding_index,
        )
        if add_start_and_stop is not None:
            self._set_token_len_fn(add_start_and_stop)
