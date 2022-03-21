import torch
import torch.nn as nn
from typing import List

from perf_gan.models.blocks.conv_blocks import ConvTransposeBlock
from perf_gan.models.d_block import DBlock
from perf_gan.models.u_block import UBlock
from perf_gan.models.bottleneck import Bottleneck


class Generator(nn.Module):
    """ Generator for performance contours modelling relying on a U-Net architecture
    """

    def __init__(self, down_channels: List[int], up_channels: List[int],
                 down_dilations: List[int], up_dilations: List[int]) -> None:
        """Initialize the generator of the performance GAN. 

        Args:
            down_channels (List[int]): list of downsampling channels
            up_channels (List[int]): list of upsampling channels
            down_dilations (List[int]): list of dilation factors for downsampling blocks
            up_dilations (List[int]): list of dilation factors for upsampling blocks
        """

        super(Generator, self).__init__()

        self.down_channels_in = down_channels[:-1]
        self.down_channels_out = down_channels[1:]
        self.down_dilations = down_dilations

        self.up_channels_in = up_channels[:-1]
        self.up_channels_out = up_channels[1:]
        self.up_dilations = up_dilations

        is_first = [True] + [False] * (len(self.down_channels_in) - 1)
        is_last = [False] * (len(self.up_channels_in) - 1) + [True]

        self.down_blocks = nn.ModuleList([
            DBlock(in_channels=in_channels,
                   out_channels=out_channels,
                   dilation=dilation,
                   first=f) for in_channels, out_channels, dilation, f in zip(
                       self.down_channels_in, self.down_channels_out,
                       self.down_dilations, is_first)
        ])

        self.up_blocks = nn.ModuleList([
            UBlock(in_channels=in_channels,
                   out_channels=out_channels,
                   dilation=dilation,
                   last=l) for in_channels, out_channels, dilation, l in zip(
                       self.up_channels_in, self.up_channels_out,
                       self.up_dilations, is_last)
        ])

        self.bottleneck = Bottleneck(in_channels=down_channels[-1],
                                     out_channels=up_channels[0])

        self.top = ConvTransposeBlock(in_channels=up_channels[-1],
                                      out_channels=up_channels[-1],
                                      dilation=1)

        # initialize weights:
        self.__initialize_weights()

    def __initialize_weights(self) -> None:
        """Initialize weights of the generator (help training)
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def down_sampling(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample the input (compute every outputs of the downsampling branch of the U-Net)

        Args:
            x (torch.Tensor): input of the downsampling branch

        Returns:
            torch.Tensor: output of the downsampling branch
        """
        for d_block in self.down_blocks:
            x = d_block(x)
        return x

    def up_sampling(self, x: torch.Tensor) -> torch.Tensor:
        """Upsampling the input (compute every outputs of the upsampling branch of the U-Net)

        Args:
            x (torch.Tensor): input of the upsampling branch

        Returns:
            torch.Tensor: output of the upsampling branch
        """
        for u_block in self.up_blocks:
            x = u_block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pass forward for the generator

        Args:
            x (torch.Tensor): input contours of size (B, 2, L)

        Returns:
            torch.Tensor: output contours of size (B, 2, L)
        """
        x = self.down_sampling(x)
        x = self.bottleneck(x)
        x = self.up_sampling(x)

        out = self.top(x)

        return out
