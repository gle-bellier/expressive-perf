import torch
import torch.nn as nn
from perf_gan.models.blocks.conv_blocks import ConvTransposeBlock
from perf_gan.utils.get_padding import get_padding


class UBlock(nn.Module):
    """Upsampling block"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dilation: int,
                 last=False, 
                 dropout=0.) -> None:
        """Initialize upsampling block

        Args:
            in_channels (int): number of input channels 
            out_channels (int): number of output channels 
            dilation (int): dilation of the convolutional block
            last (bool, optional): set to True if last block of the upsampling branch. Defaults to False.
        """
        super().__init__()
        self.last = last
        if not last:
            self.main = nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvTransposeBlock(in_channels,
                                   out_channels,
                                   dilation=dilation,
                                   norm=True, 
                                   dropout=dropout),
                ConvTransposeBlock(out_channels,
                                   out_channels,
                                   dilation=dilation,
                                   norm=True, 
                                   dropout=dropout))

            self.residual = nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvTransposeBlock(in_channels=in_channels,
                                   out_channels=out_channels,
                                   dilation=dilation,
                                   norm=True, 
                                   dropout=dropout))
        else:
            self.residual = nn.Sequential(
                nn.ConvTranspose1d(in_channels,
                                   out_channels,
                                   kernel_size=3,
                                   padding=get_padding(3, 1, dilation),
                                   dilation=dilation,
                                   norm=True, 
                                   dropout=dropout))

        self.gru = nn.GRU(out_channels, out_channels, batch_first=True, dropout=dropout)
        self.lr = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pass forward for the Upsampling convolution block.

        Args:
            x (torch.Tensor): block input of size (B, C_in, L)

        Returns:
            torch.Tensor: computed output of size (B, C_out, L)
        """

        residual = self.residual(x)

        if self.last:
            main = 0
        else:
            main = self.main(x)

        x = main + residual

        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x.permute(0, 2, 1)

        return self.lr(x)
