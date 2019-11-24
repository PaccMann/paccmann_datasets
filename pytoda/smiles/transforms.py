"""SMILES transforms."""
import re
import warnings

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from selfies import encoder

from ..transforms import Transform
from ..types import Indexes
from .smiles_language import SMILESLanguage


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


class LeftPadding(Transform):
    """Left pad token indexes."""

    def __init__(self, padding_length: int, padding_index: int) -> None:
        """
        Initialize a left padding token indexes object.

        Args:
            padding_length (int): length of the padding.
            padding_index (int): padding index.
        """
        self.padding_length = padding_length
        self.padding_index = padding_index

    def __call__(self, token_indexes: Indexes) -> Indexes:
        """
        Apply the transform.

        Args:
            token_indexes (Indexes): token indexes.

        Returns:
            Indexes: left padded indexes representation.
        """
        return (self.padding_length -
                len(token_indexes)) * [self.padding_index] + token_indexes


class ToTensor(Transform):
    """Transform token indexes to torch tensor."""

    def __init__(self, device, dtype=torch.short) -> None:
        """
        Initialize a token indexes to tensor object.

        Args:
            dtype (torch.dtype): data type. Defaults to torch.short.
            device (torch.device): device where the tensors are stored.
            Defaults to gpu, if available.
        """
        self.dtype = torch.short
        self.device = device

    def __call__(self, token_indexes: Indexes) -> torch.Tensor:
        """
        Apply the transform.

        Args:
            token_indexes (Indexes): token indexes.

        Returns:
            torch.Tensor: tensor representation of the token indexes.
        """
        return torch.tensor(
            token_indexes, dtype=self.dtype, device=self.device
        ).view(-1, 1)


class RemoveIsomery(Transform):
    """ Remove isomery (isotopic and chiral specifications) from SMILES """

    def __init__(self, bonddir=True, chirality=True) -> None:
        """
        Keyword Arguments:
            bonddir (bool): whether bond direction information should be
                removed or not (default: {True})
            chirality (bool): whether chirality information should be removed
                (default: {True}).
        """

        self.chirality_dict = {'[': '', ']': '', '@': '', 'H': ''}
        self.bond_dict = {'/': '', '\\': ''}
        self.charge = re.compile(r'\[\w+[\+\-]\d?\]')
        self.multichar_atom = re.compile(r'\[[A-Z][a-z]\w?[0-9]?\]')
        self.bonddir = bonddir
        self.chirality = chirality

    def __call__(self, smiles: str) -> str:
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

        if not self.bonddir and not self.chirality:
            return smiles
        elif self.bonddir and not self.chirality:
            update_dict = self.bond_dict
        elif self.chirality and not self.bonddir:
            update_dict = self.chirality_dict
        else:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol, isomericSmiles=False)

        updates = str.maketrans(update_dict)

        protect = []
        for m in self.charge.finditer(smiles):
            list_charg = list(range(m.start(), m.end()))
            protect += list_charg

        for m in self.multichar_atom.finditer(smiles):
            list_mc = list(range(m.start(), m.end()))
            protect += list_mc

        new_str = []
        smiles = smiles.replace('[nH]', 'N')
        for index, i in enumerate(smiles):
            new = i.translate(updates) if index not in protect else i
            new_str += list(new)
        smiles = ''.join(new_str).replace('[n]', '[nH]').replace('[N]', '[NH]')
        try:
            Chem.SanitizeMol(Chem.MolFromSmiles(smiles, sanitize=False))
            return smiles
        except TypeError:
            warnings.warn(f'Invalid SMILES {smiles}')
            return ''


class Kekulize(Transform):
    """ Transform SMILES to Kekule version """

    def __init__(self, allBondsExplicit=False, allHsExplicit=False):

        # NOTE: Explicit bonds or Hs without Kekulization is not supported
        self.allBondsExplicit = allBondsExplicit
        self.allHsExplicit = allHsExplicit

    def __call__(self, smiles: str) -> str:
        """
        Apply the kekulization transform.

        Args:
            smiles (str): a SMILES representation.
            allBondsExplicit (bool): whether bonds are explicitly encoded.

        Returns:
            str: Kekulized SMILES of same molecule.
        """
        molecule = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(molecule)
        return Chem.MolToSmiles(
            molecule,
            kekuleSmiles=True,
            allBondsExplicit=self.allBondsExplicit,
            allHsExplicit=self.allHsExplicit
        )


class Augment(Transform):
    """Augment a SMILES string, according to Bjerrum (2017)."""

    def __init__(
        self, kekuleSmiles=False, allBondsExplicit=False, allHsExplicit=False
    ) -> None:
        """ NOTE:  These parameter need to be passed down to the enumerator."""

        self.kekuleSmiles = kekuleSmiles
        self.allBondsExplicit = allBondsExplicit
        self.allHsExplicit = allHsExplicit

    def __call__(self, smiles: str) -> str:
        """
        Apply the transform.

        Args:
            smiles (str): a SMILES representation.

        Returns:
            str: randomized SMILES representation.
        """
        molecule = Chem.MolFromSmiles(smiles)
        atom_indexes = list(range(molecule.GetNumAtoms()))
        if len(atom_indexes) == 0:  # RDkit error handling
            return smiles
        np.random.shuffle(atom_indexes)
        renumbered_molecule = Chem.RenumberAtoms(molecule, atom_indexes)
        if self.kekuleSmiles:
            Chem.Kekulize(renumbered_molecule)

        return Chem.MolToSmiles(
            renumbered_molecule,
            canonical=False,
            kekuleSmiles=self.kekuleSmiles,
            allBondsExplicit=self.allBondsExplicit,
            allHsExplicit=self.allHsExplicit
        )


class Randomize(Transform):
    """ Randomize a molecule by truly shuffling all tokens. """

    def __call__(self, tokens: Indexes) -> Indexes:
        """
        NOTE: Must not apply this transformation on SMILES string, only on the
            tokenized, numerical vectors (i.e. after SMILESToTokenIndexes)

        Arguments:
            Tokens: indexes representation for the SMILES to be randomized.
        Returns:
           Indexes: shuffled indexes representation of the molecule
        """
        np.random.shuffle(tokens)
        return tokens


class Selfies(Transform):
    """ Convert a molecule from SMILES to SELFIES. """

    def __call__(self, smiles: str) -> str:
        return encoder(smiles)


class SMILESToMorganFingerprints(Transform):
    """Get fingerprints starting from SMILES."""

    def __init__(self, radius: int = 2, bits: int = 512) -> None:
        """
        Initialize a SMILES to fingerprints object.

        Args:
            radius (int): radius of the fingerprints.
            bits (int): bits used to represent the fingerprints.
        """
        self.radius = radius
        self.bits = bits

    def __call__(self, smiles: str) -> np.array:
        """
        Apply the transform.

        Args:
            smiles (str): a SMILES representation.

        Returns:
            np.array: the fingerprints.
        """
        try:
            molecule = Chem.MolFromSmiles(smiles)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                molecule, self.radius, nBits=self.bits
            )
        except Exception:
            warnings.warn(f'Invalid SMILES {smiles}')
            molecule = Chem.MolFromSmiles('')
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                molecule, self.radius, nBits=self.bits
            )
        array = np.zeros((1, ))
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array
