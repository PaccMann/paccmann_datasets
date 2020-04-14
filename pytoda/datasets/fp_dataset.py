"""FPDataset module."""
import torch
from torch.utils.data import Dataset
from ..smiles.transforms import SMILESToMorganFingerprints, ToTensor
from ..types import FileList
from ..transforms import Compose


class FPDataset(Dataset):
    """
    Fingerprint dataset implementation.
    """

    def __init__(
        self,
        *smi_filepaths: FileList,
        chirality: bool = True,
        radius: int = 2,
        bits: int = 256,
        device: torch.device = torch.
        device('cuda' if torch.cuda.is_available() else 'cpu'),
        backend: str = 'eager'
    ) -> None:
        """
        Initialize a Fingerprint dataset (uses ECFP).

        Args:
            smi_filepaths (FileList): paths to .smi files.
            chirality (bool): Whether chirality is considered to get FP.
            radius (int): Radius to consider to compute FP.
            bits (int): number of bits of ECFP.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
            backend (str): memeory management backend.
                Defaults to eager, prefer speed over memory consumption.
        """
        Dataset.__init__(self)
        transforms = [
            SMILESToMorganFingerprints(
                radius=radius, bits=bits, chirality=chirality
            )
        ]
        transforms += [ToTensor(device=self.device)]
        self.transform = Compose(transforms)

    def __len__(self) -> int:
        """Total number of samples."""
        return len(self._dataset)

    def __getitem__(self, index: int) -> torch.tensor:
        """
        Generates one sample of data.

        Args:
            index (int): index of the sample to fetch.

        Returns:
            torch.tensor: a torch tensor of token indexes,
                for the current sample.
        """
        return self._dataset[index]
