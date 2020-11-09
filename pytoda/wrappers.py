import torch
import torch.nn as nn
import neuralnet_pytorch.metrics


class WrapperCDist(nn.Module):
    """Wrapper for torch.cdist module for easy argument passing."""

    def __init__(self, p: int = 2) -> None:
        """Constructor.

        Args:
            p (int, optional): p value for the p-norm distance to calculate
                between each vector pair. Defaults to 2.
        """

        super(WrapperCDist, self).__init__()
        self.p = p

    def forward(self, set1: torch.Tensor, set2: torch.Tensor) -> torch.Tensor:
        """Computes the pairwise p-norms.

        Args:
            set1 (torch.Tensor): Input tensor of shape
                [batch_size, length1, dim]
            set2 (torch.Tensor): Input tensor of shape
                [batch_size, length2, dim]

        Returns:
            torch.Tensor: Tensor of shape [batch_size, length1, length2]
                representing the pairwise distances.
        """
        return torch.cdist(set1, set2, self.p)


class WrapperEMD(nn.Module):
    """Wrapper for Earth Mover's Distance for easy argument passing."""

    def __init__(
        self, reduction: str = 'mean', sinkhorn: bool = False
    ) -> None:
        """Constructor.

        Args:
            reduction (str, optional): One of 'mean' or 'sum'.
                Defaults to 'mean'.
            sinkhorn (bool, optional): whether to use the Sinkhorn approximation
                of the Wasserstein distance. False will fall back to a CUDA
                implementation, which is only available if the CUDA-extended
                neuralnet-pytorch is installed. Defaults to False.
        """

        super(WrapperEMD, self).__init__()

        self.reduction = reduction
        self.sinkhorn = sinkhorn

    def forward(self, set1: torch.Tensor, set2: torch.Tensor) -> torch.Tensor:
        """Computes the pairwise Wasserstein/Earth Mover's Distance.

        Args:
            set1 (torch.Tensor): Input tensor of shape
                [batch_size, length1, dim]
            set2 (torch.Tensor): Input tensor of shape
                [batch_size, length2, dim]

        Returns:
            torch.Tensor: Tensor of shape [batch_size, length1, length2]
                representing the pairwise distances.
        """

        return neuralnet_pytorch.metrics.emd_loss(
            set1, set2, self.reduction, self.sinkhorn
        )


class WrapperKLDiv(nn.Module):
    """Wrapper for KL-Divergence for easy argument passing."""

    def __init__(self, reduction: str = 'mean') -> None:
        """Constructor.

        Args:
            reduction (str, optional): One of 'none','batchmean','sum', 'mean'.
                Defaults to 'mean'.
        """

        super(WrapperKLDiv, self).__init__()

        self.reduction = reduction

    def forward(self, set1: torch.Tensor, set2: torch.Tensor) -> torch.Tensor:
        """Computes the KL-Divergence.

        Args:
            set1 (torch.Tensor): Input tensor of arbitrary shape.
            set2 (torch.Tensor): Tensor of the same shape as input.

        Returns:
            torch.Tensor: Scalar by default. if reduction = 'none', then same
                shape as input.
        """

        return nn.functional.kl_div(set1, set2, reduction=self.reduction)
