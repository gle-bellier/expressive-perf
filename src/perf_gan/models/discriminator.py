import torch
import torch.nn as nn
from typing import List

from perf_gan.models.conv_blocks import ConvBlock


class Discriminator(nn.Module):
    """Discriminator for performance contours modelling relying on a U-Net architecture
    """
    def __init__(self, conv_channels: List[int], dilations: List[int],
                 h_dims: List[int]) -> None:
        """Initialize the discriminator of the performance GAN. 

        Args:
            conv_channels (List[int]): channels of each convolutional block
            dilations (List[int]): list of dilations of convolutional block
            h_dims (List[int]): dims of linear layers
        """

        super(Discriminator, self).__init__()

        self.conv = nn.ModuleList([
            ConvBlock(in_channels=in_channels,
                      out_channels=out_channels,
                      dilation=dilation)
            for in_channels, out_channels, dilation in zip(
                conv_channels[:-1], conv_channels[1:], dilations)
        ])
        self.linears = nn.ModuleList([
            nn.Linear(in_features=in_features, out_features=out_features)
            for in_features, out_features in zip(h_dims[:-1], h_dims[1:])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pass forward

        Args:
            x (torch.Tensor): input contours of size (B, C, L)

        Returns:
            torch.Tensor: output tensor of size (B, 1, 1)
        """
        for conv in self.conv:
            x = conv(x)
        for l in self.linears:
            x = l(x)
        return torch.sigmoid(x)
