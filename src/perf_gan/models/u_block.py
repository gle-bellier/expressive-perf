import torch
import torch.nn as nn
from perf_gan.models.blocks.conv_blocks import ConvTransposeBlock


class UBlock(nn.Module):
    """Upsampling block"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 upsample=False,
                 dropout=0.) -> None:

        super().__init__()
        self.main = nn.Sequential(
            ConvTransposeBlock(in_channels,
                               out_channels,
                               upsample=upsample,
                               dropout=dropout),
            ConvTransposeBlock(out_channels,
                               out_channels,
                               upsample=False,
                               dropout=dropout))

        self.residual = ConvTransposeBlock(in_channels,
                                           out_channels,
                                           upsample=upsample,
                                           dropout=dropout)

        self.top = ConvTransposeBlock(out_channels,
                                      out_channels,
                                      upsample=False,
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
        x = self.lr(x)
        main = self.main(x)

        out = self.top(res + main)
        return out
