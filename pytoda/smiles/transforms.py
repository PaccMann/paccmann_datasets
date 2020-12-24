"""SMILES transforms."""
import logging
import re

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from selfies import encoder
import torch

import pytoda

from ..transforms import Transform
from ..types import Indexes
from .smiles_language import SMILESLanguage

logger = logging.getLogger('pytoda_SMILES_transforms')


class SMILESToTokenIndexes(Transform):
    """Transform SMILES to token indexes using SMILES language."""

    def __init__(self, smiles_language: SMILESLanguage) -> None:
        """
        Initialize a SMILES to token indexes object.

        Args:
            smiles_language (SMILESLanguage): a SMILES language.
        """
        self.smiles_language = smiles_language

    def __call__(self, smiles: str) -> Indexes:
        """
        Apply the SMILES tokenization transformation

        Args:
            smiles (str): a SMILES representation.

        Returns:
            Indexes: indexes representation for the SMILES provided.
        """
        return self.smiles_language.smiles_to_token_indexes(smiles)


class RemoveIsomery(Transform):
    """ Remove isomery (isotopic and chiral specifications) from SMILES """

    def __init__(self, bonddir=True, chirality=True, sanitize=True) -> None:
        """
        Initialize isomery removal.

        Args:
            bonddir (bool): whether bond direction information should be
                removed or not (default: {True})
            chirality (bool): whether chirality information should be removed
                (default: {True}).
        """

        self.chirality_dict = {'[': '', ']': '', '@': '', 'H': ''}
        self.bond_dict = {'/': '', '\\': ''}
        self.charge = re.compile(r'\[\w+\@?\@?[\+\-]\d?\]')
        self.multichar_atom = re.compile(r'\[[0-9]?[A-Za-z][a-z]?\w?[2-8]?\]')
        self.bonddir = bonddir
        self.chirality = chirality
        self.sanitize = sanitize

        if not self.bonddir and not self.chirality:
            self._call_fn = lambda smiles: smiles
        elif self.bonddir and not self.chirality:
            self.updates = str.maketrans(self.bond_dict)
            self._call_fn = self._isomery_call_fn
        elif self.chirality and not self.bonddir:
            self.updates = str.maketrans(self.chirality_dict)
            self._call_fn = self._isomery_call_fn
        else:
            self._call_fn = lambda smiles: Chem.MolToSmiles(
                Chem.MolFromSmiles(smiles, sanitize=sanitize),
                isomericSmiles=False,
                canonical=False
            )

    def _isomery_call_fn(self, smiles: str) -> str:
        """
        Remove the stereoinfo of the SMILES. That can either be the removal of
        only the chirality information (at tetrahedral carbons with four
        different substituents) and/or the removal of the bond direction
        information (either the highest CIP-rated substituents on the same side
        of the two carbons of the double bond (Z) or on differnet sides (E)).

        NOTE: If bonddir=True and chirality=True, no transformation is applied.

        Args:
            smiles (str): a SMILES representation.

        Returns:
            str: SMILES representation of original smiles string with removed
                stereoinfo checked for validity
        """
        smiles = smiles.replace('[nH]', 'N')
        protect = []
        for m in self.charge.finditer(smiles):
            list_charg = list(range(m.start(), m.end()))
            protect += list_charg

        for m in self.multichar_atom.finditer(smiles):
            list_mc = list(range(m.start(), m.end()))
            protect += list_mc

        new_str = []
        for index, i in enumerate(smiles):
            new = i.translate(self.updates) if index not in protect else i
            new_str += list(new)
        smiles = ''.join(new_str).replace('N@@', 'N').replace('N@', 'N')

        try:
            Chem.SanitizeMol(
                Chem.MolFromSmiles(smiles, sanitize=self.sanitize)
            )
            return smiles
        except TypeError:
            logger.warning(f'\nInvalid SMILES {smiles}')
            return ''

    def __call__(self, smiles: str) -> str:
        """
        Executable of RemoveIsomery class. The _call_fn is determined in
        the constructor based on the bonddir and chirality parameter.

        Args:
            smiles (str): a SMILES sequence.

        Returns:
            str: SMILES after the _call_fn was applied.
        """

        return self._call_fn(smiles)


class Kekulize(Transform):
    """Transform SMILES to Kekule version."""

    def __init__(
        self, all_bonds_explicit=False, all_hs_explicit=False, sanitize=True
    ):

        # NOTE: Explicit bonds or Hs without Kekulization is not supported
        self.all_bonds_explicit = all_bonds_explicit
        self.all_hs_explicit = all_hs_explicit
        self.sanitize = sanitize

    def __call__(self, smiles: str) -> str:
        """
        Apply the kekulization transform.

        Args:
            smiles (str): a SMILES representation.
            all_bonds_explicit (bool): whether bonds are explicitly encoded.

        Returns:
            str: Kekulized SMILES of same molecule.
        """
        try:
            molecule = Chem.MolFromSmiles(smiles, sanitize=self.sanitize)
            if not self.sanitize:
                # Properties as valence are not calculated if sanitize is False
                molecule.UpdatePropertyCache(strict=False)
            Chem.Kekulize(molecule)
            return Chem.MolToSmiles(
                molecule,
                kekuleSmiles=True,
                allBondsExplicit=self.all_bonds_explicit,
                allHsExplicit=self.all_hs_explicit,
                canonical=False
            )
        except Exception:
            logger.warning(
                f'\nInvalid SMILES {smiles}, no kekulization done '
                f'bondsExplicit: {self.all_bonds_explicit} & HsExplicit: '
                f'{self.all_hs_explicit} are also ignored.'
            )
            return smiles


class NotKekulize(Transform):
    """ Transform SMILES without explicitly converting to Kekule version """

    def __init__(
        self, all_bonds_explicit=False, all_hs_explicit=False, sanitize=True
    ):
        self.all_bonds_explicit = all_bonds_explicit
        self.all_hs_explicit = all_hs_explicit
        self.sanitize = sanitize

    def __call__(self, smiles: str) -> str:
        """
        Apply transform.

        Args:
            smiles (str): a SMILES representation.

        Returns:
            str: SMILES of same molecule.
        """
        try:
            molecule = Chem.MolFromSmiles(smiles, sanitize=self.sanitize)
            return Chem.MolToSmiles(
                molecule,
                allBondsExplicit=self.all_bonds_explicit,
                allHsExplicit=self.all_hs_explicit,
                canonical=False
            )
        except Exception:
            logger.warning(
                f'\nInvalid SMILES {smiles}, HsExplicit:{self.all_hs_explicit}'
                f'and bondsExplicit: {self.all_bonds_explicit} are ignored.'
            )
            return smiles


class Augment(Transform):
    """Augment a SMILES string, according to Bjerrum (2017)."""

    def __init__(
        self,
        kekule_smiles: bool = False,
        all_bonds_explicit: bool = False,
        all_hs_explicit: bool = False,
        sanitize: bool = True,
        seed: int = -1
    ) -> None:
        """ NOTE:  These parameter need to be passed down to the enumerator."""

        self.kekule_smiles = kekule_smiles
        self.all_bonds_explicit = all_bonds_explicit
        self.all_hs_explicit = all_hs_explicit
        self.sanitize = sanitize
        self.seed = seed
        if self.seed > -1:
            np.random.seed(self.seed)

    def __call__(self, smiles: str) -> str:
        """
        Apply the transform.

        Args:
            smiles (str): a SMILES representation.

        Returns:
            str: randomized SMILES representation.
        """
        molecule = Chem.MolFromSmiles(smiles, sanitize=self.sanitize)
        if molecule is None:
            logger.warning(f'\nAugmentation skipped for invalid mol: {smiles}')
            return smiles
        if not self.sanitize:
            molecule.UpdatePropertyCache(strict=False)
        atom_indexes = list(range(molecule.GetNumAtoms()))
        if len(atom_indexes) == 0:  # RDkit error handling
            return smiles
        np.random.shuffle(atom_indexes)
        renumbered_molecule = Chem.RenumberAtoms(molecule, atom_indexes)
        if self.kekule_smiles:
            Chem.Kekulize(renumbered_molecule)

        return Chem.MolToSmiles(
            renumbered_molecule,
            canonical=False,
            kekuleSmiles=self.kekule_smiles,
            allBondsExplicit=self.all_bonds_explicit,
            allHsExplicit=self.all_hs_explicit
        )


class AugmentTensor(Transform):
    """
    Augment a SMILES (represented as a Tensor) according to Bjerrum (2017).
    """

    def __init__(
        self,
        smiles_language,
        kekule_smiles=False,
        all_bonds_explicit=False,
        all_hs_explicit=False,
        sanitize=True
    ) -> None:
        """NOTE:  These parameter need to be passed down to the enumerator."""
        self.smiles_language = smiles_language
        self.kekule_smiles = kekule_smiles
        self.all_bonds_explicit = all_bonds_explicit
        self.all_hs_explicit = all_hs_explicit
        self.sanitize = sanitize

    def update_smiles_language(self, smiles_language):
        if not isinstance(
            smiles_language, pytoda.smiles.smiles_language.SMILESLanguage
        ):
            raise ValueError('Please pass a SMILES language object')
        self.smiles_language = smiles_language

    def __call__(self, smiles_numerical: list) -> str:
        """
        Apply the transform.

        Args:
            smiles_numerical (Union[list, torch.Tensor]): either a SMILES
                represented as list of ints or a Tensor.

        Returns:
            torch.Tensor: randomized SMILES representation.
        """

        if type(smiles_numerical) == list:
            smiles = self.smiles_language.token_indexes_to_smiles(
                smiles_numerical
            )
            try:
                molecule = Chem.MolFromSmiles(smiles, sanitize=self.sanitize)
                atom_indexes = list(range(molecule.GetNumAtoms()))
                if len(atom_indexes) == 0:  # RDkit error handling
                    return smiles
                np.random.shuffle(atom_indexes)
                renumbered_molecule = Chem.RenumberAtoms(
                    molecule, atom_indexes
                )
                if self.kekule_smiles:
                    Chem.Kekulize(renumbered_molecule)

                augmented_smiles = Chem.MolToSmiles(
                    renumbered_molecule,
                    canonical=False,
                    kekuleSmiles=self.kekule_smiles,
                    allBondsExplicit=self.all_bonds_explicit,
                    allHsExplicit=self.all_hs_explicit
                )
            except Exception:
                logger.warning(
                    f'\nAugmentation skipped, mol invalid: {smiles}'
                )
                augmented_smiles = smiles
            return self.smiles_language.smiles_to_token_indexes(
                augmented_smiles
            )
        elif type(smiles_numerical) == torch.Tensor:
            return self.__call__tensor(smiles_numerical)

        else:
            raise TypeError('Please pass either a torch.Tensor or a list.')

    def __call__tensor(self, smiles_numerical: torch.Tensor) -> str:
        """
        Wrapper of the transform for torch.Tensor.

        Args:
            smiles_numerical (torch.Tensor): a Tensor with SMILES represented
                as ints. Needs to have shape batch_size x sequence_length.
        Returns:
            str: randomized SMILES representation.
        """

        # Infer the padding type to ensure returning tensor of same shape.
        if self.smiles_language.padding_index in smiles_numerical.flatten():

            padding = True
            left_padding = any([
                self.smiles_language.padding_index == row[0]
                for row in smiles_numerical
            ])  # yapf: disable
            right_padding = any([
                self.smiles_language.padding_index == row[-1]
                for row in smiles_numerical
            ])  # yapf: disable
            if (
                (left_padding and right_padding)
                or (not left_padding and not right_padding)
            ):
                raise ValueError(
                    'Could not uniqely infer padding type. Leftpadding was '
                    f'{left_padding}, right_padding was {right_padding}.'
                )
        else:
            padding = False

        seq_len = smiles_numerical.shape[1]

        # Loop over tensor (SMILES by SMILES) and augment. Exclude augmentation
        # if it violates the padding
        augmented = []
        for smiles in smiles_numerical:

            lenx = seq_len + 1
            while lenx > seq_len:
                augmented_smiles = self.__call__(smiles.tolist())
                lenx = len(augmented_smiles)
            if padding:
                pads = (
                    [self.smiles_language.padding_index] *
                    (seq_len - len(augmented_smiles))
                )
                if right_padding:
                    augmented_smiles = augmented_smiles + pads
                if left_padding:
                    augmented_smiles = pads + augmented_smiles

            augmented.append(
                torch.unsqueeze(torch.Tensor(augmented_smiles), 0)
            )

        augmented = torch.cat(augmented, dim=0)
        return augmented


class Selfies(Transform):
    """ Convert a molecule from SMILES to SELFIES. """

    def __call__(self, smiles: str) -> str:
        return encoder(smiles)


class Canonicalization(Transform):
    """
    Convert any SMILES to RDKit-canonical SMILES.
    An example:
     - input: 'CN2C(=O)N(C)C(=O)C1=C2N=CN1C'
     - output: 'Cn1c(=O)c2c(ncn2C)n(C)c1=O'
    """

    def __init__(self, sanitize=True):
        self.sanitize = sanitize

    def __call__(self, smiles: str) -> str:
        try:
            canon = Chem.MolToSmiles(
                Chem.MolFromSmiles(smiles, sanitize=self.sanitize),
                canonical=True
            )
            return canon
        except Exception:
            logger.warning(
                f'\nInvalid SMILES {smiles}, no canonicalization done'
            )
            return smiles


class SMILESToMorganFingerprints(Transform):
    """Get fingerprints starting from SMILES."""

    def __init__(
        self, radius: int = 2, bits: int = 512, chirality=True, sanitize=False
    ) -> None:
        """
        Initialize a SMILES to fingerprints object.

        Args:
            radius (int): radius of the fingerprints.
            bits (int): bits used to represent the fingerprints.
        """
        self.radius = radius
        self.bits = bits
        self.chirality = chirality
        self.sanitize = sanitize

    def __call__(self, smiles: str) -> np.array:
        """
        Apply the transform.

        Args:
            smiles (str): a SMILES representation.

        Returns:
            np.array: the fingerprints.
        """
        try:
            molecule = Chem.MolFromSmiles(smiles, sanitize=self.sanitize)
            if not self.sanitize:
                molecule.UpdatePropertyCache(strict=False)
                AllChem.FastFindRings(molecule)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                molecule,
                self.radius,
                nBits=self.bits,
                useChirality=self.chirality
            )
        except Exception:
            logger.warning(f'\nInvalid SMILES {smiles}')
            molecule = Chem.MolFromSmiles('')
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                molecule, self.radius, nBits=self.bits
            )
        array = np.zeros((1, ))
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array
