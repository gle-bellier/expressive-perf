import torch
import torch.nn as nn
from typing import List

from perf_gan.models.blocks.conv_blocks import ConvBlock


class LinBlock(nn.Module):
    """Linear block
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 norm=False) -> None:
        super(LinBlock, self).__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.lr = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass for the linear block

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """

        x = self.linear(x)
        return self.lr(x)