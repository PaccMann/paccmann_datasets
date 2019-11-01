"""SMILES transforms."""
import torch
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from ..transforms import Transform
from .smiles_language import SMILESLanguage
from ..types import Indexes


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
        Apply the transform.

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
            device (torch.device): device where the tensors are stored. Defaults to gpu, if available.
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


class SMILESRandomization(Transform):
    """Randomize a SMILES."""

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
        np.random.shuffle(atom_indexes)
        renumbered_molecule = Chem.RenumberAtoms(molecule, atom_indexes)
        return Chem.MolToSmiles(renumbered_molecule, canonical=False)


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
            fingeprints = AllChem.GetMorganFingerprintAsBitVect(
                molecule, self.radius, nBits=self.bits
            )
        except Exception:
            molecule = Chem.MolFromSmiles('')
            fingeprints = AllChem.GetMorganFingerprintAsBitVect(
                molecule, self.radius, nBits=self.bits
            )
        array = np.zeros((1, ))
        DataStructs.ConvertToNumpyArray(fingeprints, array)
        return array
