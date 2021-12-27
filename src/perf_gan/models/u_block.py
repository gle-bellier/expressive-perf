import torch
import torch.nn as nn
from perf_gan.models.conv_blocks import ConvBlock


class UBlock(nn.Module):
    """Upsampling block"""
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dilation: int,
                 last=False) -> None:
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
                ConvBlock(in_channels, out_channels, dilation=dilation),
                ConvBlock(out_channels, out_channels, dilation=dilation))

            self.residual = nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dilation=dilation,
                ))
        else:
            self.main = nn.Sequential(
                ConvBlock(in_channels, out_channels, dilation=dilation),
                ConvBlock(out_channels, out_channels, dilation=dilation))

            self.residual = nn.Sequential(
                ConvBlock(in_channels, out_channels, dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pass forward for the Upsampling convolution block.

        Args:
            x (torch.Tensor): block input of size (B, C_in, L)

        Returns:
            torch.Tensor: computed output of size (B, C_out, L)
        """
        main = self.main(x)
        residual = self.residual(x)
        return main + residual
