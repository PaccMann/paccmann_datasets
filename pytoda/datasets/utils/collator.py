import numpy as np
from typing import Tuple, List
import torch
from torch.utils.data._utils.collate import default_convert
from pytoda.datasets.utils.factories import BACKGROUND_TENSOR_FACTORY


class Collator:
    """Contains function to pad data returned by dataloader."""

    def __init__(
        self,
        padding_mode: List,
        padding_values: List,
        dim: int,
        max_length: int,
        batch_first: bool = True,
        device: torch.device = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ),
    ):
        """Constructor.

        Args:
            dim (int): Dimension of the data.
            max_length (int): Maximum set length.
            batch_first (bool, optional): Whether batch size is the first
                dimension or not. Defaults to True.
            device (torch.device): device where the tensors are stored.
                Defaults to gpu, if available.
        """
        super(Collator, self).__init__()
        # self.background_tensor = super().constant_value_tensor()
        self.padding_mode = padding_mode
        self.padding_values = padding_values
        # self.dim = dim
        self.max_len = max_length
        self.batch_first = batch_first
        self.device = device

    def __call__(self, DataLoaderBatch: Tuple) -> Tuple:
        """Collate function for batch-wise padding of samples.

        Args:
            DataLoaderBatch (Tuple): Tuple of tensors returned by get_item of the
            dataset class.

        Returns:
            Tuple: Tuple of padded input tensors and tensor of set lengths.
        """

        batch_size = len(DataLoaderBatch)
        # convert [(a,b,c),(e,f,g)] to [[a,e],[b,f],[c,g]]
        batch_split = list(zip(*DataLoaderBatch))

        returned_tensors = []

        for index, batch in enumerate(batch_split):

            if self.padding_mode == "none":
                returned_tensors.append(batch)
            else:

                mode = self.padding_mode[index]
                value = self.padding_values
                trailing_dims = batch.size()[2:]

                if self.batch_first:
                    out_dims = (batch_size, self.max_len, trailing_dims)
                else:
                    out_dims = (self.max_len, batch_size, trailing_dims)

                out_tensor = BACKGROUND_TENSOR_FACTORY[mode](
                    value, out_dims, device=self.device
                )

                for datum_index, tensor in enumerate(default_convert(batch)):

                    length = tensor.size(0)

                    if self.batch_first:
                        out_tensor[datum_index, :length, ...] = tensor
                    else:
                        out_tensor[:length, datum_index, ...] = tensor

                returned_tensors.append(out_tensor)

        return tuple(returned_tensors)

        # sets1, sets2, targs12, targs21 = (
        #     batch_split[0],
        #     batch_split[1],
        #     batch_split[2],
        #     batch_split[3],
        # )

        # lengths = list(map(len, sets1))

        # padded_sets1 = torch.full(
        #     (batch_size, self.max_len, self.dim), self.padding_token, device=self.device
        # )
        # padded_sets2 = torch.full(
        #     (batch_size, self.max_len, self.dim), self.padding_token, device=self.device
        # )
        # targets12 = np.tile(np.arange(self.max_len), (batch_size, 1))
        # targets21 = np.tile(np.arange(self.max_len), (batch_size, 1))
        # targets12 = torch.from_numpy(targets12).to(self.device)
        # targets21 = torch.from_numpy(targets21).to(self.device)

        # for i, l in enumerate(lengths):
        #     padded_sets1[i, 0:l, :] = sets1[i][0:l, :]
        #     padded_sets2[i, 0:l, :] = sets2[i][0:l, :]

        #     targets12[i, 0:l] = targs12[i][:]
        #     targets21[i, 0:l] = targs21[i][:]

        # if self.batch_first is False:
        #     padded_sets1, padded_sets2 = (
        #         padded_sets1.permute(1, 0, 2),
        #         padded_sets2.permute(1, 0, 2),
        #     )

        # return padded_sets1, padded_sets2, targets12, targets21, torch.tensor(lengths)

