"""Polymer language handling."""
from typing import Iterable
from .smiles_language import SMILESEncoder
from .transforms import compose_smiles_transforms, compose_encoding_transforms
from ..types import Indexes, delegate_kwargs


# @delegate_kwargs
class PolymerEncoder(SMILESEncoder):
    """
    PolymerEncoder class.

    PolymerEncoder is an extension of SMILESEncoder adding special start and
    stop tokens per entity.
    A polymer language is usually shared across several SMILES datasets (e.g.
    different entity sources).
    """

    def __init__(
        self,
        entity_names: Iterable[str],
        name: str = 'polymer-language',
        add_start_and_stop: bool = True,
        **kwargs
    ) -> None:
        """
        Initialize Polymer language able to encode different entities.

        Args:
            entity_names (Iterable[str]): A list of entity names that the
                polymer language can distinguish.
            name (str): name of the PolymerEncoder.
            add_start_and_stop (bool): add start and stop token indexes.
                Defaults to True.
            kwargs (dict): additional parameters passed to SMILESEncoder.

        NOTE:
            See `set_smiles_transforms` and `set_encoding_transforms` to change
            the transforms temporarily and reset with
            `reset_initial_transforms`. Assignment of class attributes
            in the parameter list will trigger such a reset.
        """

        super().__init__(
            name=name,
            add_start_and_stop=add_start_and_stop,
            **kwargs
        )
        self.entities = list(map(lambda x: x.capitalize(), entity_names))
        self.current_entity = None

        # rebuild basic vocab to group special tokens
        # required for `token_indexes_to_smiles`
        self.start_entity_tokens, self.stop_entity_tokens = (
            list(map(lambda x: '<' + x.upper() + '_' + s + '>', entity_names))
            for s in ['START', 'STOP']
        )

        self.index_to_token = {
            self.padding_index: self.padding_token,
            self.unknown_index: self.unknown_token,
            self.start_index: self.start_token,
            self.stop_index: self.stop_token,
        }
        # NOTE: include augmentation characters, paranthesis and numbers for
        #    rings
        additional_indexes_to_token = dict(
            enumerate(
                self.start_entity_tokens + self.stop_entity_tokens +
                list('()') + list(map(str, range(1, 10))) +
                list('%{}'.format(index) for index in range(10, 30)),
                start=len(self.index_to_token)
            )
        )
        self.index_to_token.update(additional_indexes_to_token)
        self.number_of_tokens = len(self.index_to_token)
        self.token_to_index = {
            token: index
            for index, token in self.index_to_token.items()
        }

        self.reset_initial_transforms()

    def _check_entity(self, entity: str) -> str:
        entity_ = entity.capitalize()
        if entity_ not in self.entities:
            raise ValueError(f'Unknown entity was given ({entity_})')
        return entity_

    def update_entity(self, entity: str) -> None:
        """
        Update the current entity of the Polymer language object

        Args:
            entity (str): a chemical entity (e.g. 'Monomer').

        Returns:
            None
        """
        self.current_entity = self._check_entity(entity)
        self.reset_initial_transforms()

    def smiles_to_token_indexes(
        self, smiles: str, entity: str = None
    ) -> Indexes:
        """
        Transform character-level SMILES into a sequence of token indexes.

        In case of add_start_stop, inserts entity specific tokens.

        Args:
            smiles (str): a SMILES (or SELFIES) representation.
            entity (str): a chemical entity (e.g. 'Monomer'). Defaults to
                None, where the current entity is used (initially the
                SMILESEncoder default).  # TODO

        Returns:
            Indexes: indexes representation for the SMILES/SELFIES provided.
        """
        if entity is None:
            # default behavior given by call to update_entity()
            entity = self.current_entity
        else:
            entity = self._check_entity(entity)

        return self.all_encoding_transforms[entity](
            [
                self.token_to_index[token] for token in
                self.smiles_tokenizer(self.transform_smiles(smiles))
                if token in self.token_to_index
            ]
        )

    def token_indexes_to_smiles(self, token_indexes: Indexes) -> str:
        """
        Transform a sequence of token indexes into SMILES, ignoring special
        tokens.

        Args:
            token_indexes (Indexes): a sequence of token indexes.

        Returns:
            str: a SMILES representation.
        """
        return ''.join(
            [
                self.index_to_token.get(token_index, '')
                for token_index in token_indexes
                # consider only valid SMILES token indexes
                if token_index > 3 + len(self.entities) * 2
            ]
        )

    def reset_initial_transforms(self):
        super().reset_initial_transforms()
        if not hasattr(self, 'entities'):  # call from base
            return
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
            if all_bonds_explicit is not None else self.all_bonds_explicit,
            all_hs_explicit=all_hs_explicit
            if all_hs_explicit is not None else self.all_hs_explicit,
            remove_bonddir=remove_bonddir
            if remove_bonddir is not None else self.remove_bonddir,
            remove_chirality=remove_chirality
            if remove_chirality is not None else self.remove_chirality,
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
        device=None,
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
            if add_start_and_stop is not None else self.add_start_and_stop,
            start_index=start_index,
            stop_index=stop_index,
            padding=padding if padding is not None else self.padding,
            padding_length=padding_length
            if padding_length is not None else self.padding_length,
            padding_index=self.padding_index,
            device=device if device is not None else self.device,
        )
        if add_start_and_stop is not None:
            self._set_token_len_fn(add_start_and_stop)
