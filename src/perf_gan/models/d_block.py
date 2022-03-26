from typing import List
import torch
import torch.nn as nn
from perf_gan.models.blocks.conv_blocks import ConvBlock


class DBlock(nn.Module):
    """Downsampling block"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pool=False,
                 dropout=0.) -> None:

        super().__init__()
        self.main = nn.Sequential(
            ConvBlock(in_channels, out_channels, pool=pool, dropout=dropout),
            ConvBlock(out_channels, out_channels, pool=False, dropout=dropout))

        self.residual = ConvBlock(in_channels,
                                  out_channels,
                                  pool=pool,
                                  dropout=dropout)

        self.top = ConvBlock(out_channels,
                             out_channels,
                             pool=False,
                             dropout=dropout)

        self.gru = nn.GRU(in_channels, in_channels, batch_first=True)
        self.lr = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pass forward for the Upsampling convolution block.

        Args:
            x (torch.Tensor): block input   of size (B, C_in, L)

        Returns:
            torch.Tensor: computed output of size (B, C_out, L)
        """

        # compute residual
        res = self.residual(x)

        # compute main
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x.permute(0, 2, 1)
        self.lr(x)
        main = self.main(x)

        out = self.top(res + main)

        return out
