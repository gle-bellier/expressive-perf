from typing import List
import torch
import torch.nn as nn
from perf_gan.models.conv_blocks import ConvBlock


class DBlock(nn.Module):
    """Down sampling block for the U-net architecture
    """
    def __init__(self, in_channels: int, out_channels: int,
                 dilation: int) -> None:
        """Initialize the down sampling block

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            dilation (int): dilation of convolutional blocks 
        """
        super().__init__()
        self.lr = nn.LeakyReLU()
        self.conv1 = ConvBlock(in_channels, out_channels, dilation)
        self.conv2 = ConvBlock(out_channels, out_channels, dilation)
        self.mp = nn.MaxPool1d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass forward of the downsampling block

        Args:
            x (torch.Tensor): input tensor of size (B, in_C, L)

        Returns:
            torch.Tensor: output tensor  of size (B, out_C, L//2)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.mp(x)
        return out