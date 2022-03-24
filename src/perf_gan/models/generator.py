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

    def __init__(self, channels: List[int], dropout=0.) -> None:
        """Initialize the generator of the performance GAN. 

        Args:
            down_channels (List[int]): list of downsampling channels
            up_channels (List[int]): list of upsampling channels
            down_dilations (List[int]): list of dilation factors for downsampling blocks
            up_dilations (List[int]): list of dilation factors for upsampling blocks
        """

        super(Generator, self).__init__()

        self.down_blocks = nn.ModuleList([
            DBlock(in_channels=in_channels,
                   out_channels=out_channels,
                   pool=True,
                   dropout=dropout)
            for in_channels, out_channels in zip(channels[:-1], channels[1:])
        ])

        self.up_blocks = nn.ModuleList([
            UBlock(in_channels=in_channels,
                   out_channels=out_channels,
                   upsample=True,
                   dropout=dropout) for in_channels, out_channels in zip(
                       channels[-1:0:-1], channels[-2::-1])
        ])

        self.bottleneck = Bottleneck(in_channels=channels[-1],
                                     out_channels=channels[-1],
                                     dropout=dropout)

        self.top = ConvTransposeBlock(in_channels=channels[0],
                                      out_channels=2)

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


if __name__ == "__main__":
    x = torch.randn(32, 2, 1024)

    g = Generator([2, 4, 8, 16])

    print(g(x).shape)