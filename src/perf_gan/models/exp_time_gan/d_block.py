from typing import List
import torch
import torch.nn as nn
from perf_gan.models.blocks.conv_blocks import ConvBlock


class DBlock(nn.Module):
    """Down sampling block for the U-net architecture
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dilation: int,
                 first=False) -> None:
        """Initialize the down sampling block

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            dilation (int): dilation of convolutional blocks 
            first (bool, optional): set to True in first block of the downsampling branch. Defaults to False.
        """
        super().__init__()

        self.first = first
        self.lr = nn.LeakyReLU()
        self.conv1 = ConvBlock(in_channels, out_channels, dilation, norm=True)
        self.conv2 = ConvBlock(out_channels, out_channels, dilation, norm=True)

        self.mp = nn.MaxPool1d(kernel_size=2)
        self.avg = nn.AvgPool1d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass forward of the downsampling block

        Args:
            x (torch.Tensor): input tensor of size (B, in_C, L)

        Returns:
            torch.Tensor: output tensor  of size (B, out_C, L//2)
        """
        x = self.conv1(x)

        if not self.first:
            x = self.avg(x)
        out = self.conv2(x)
        return out