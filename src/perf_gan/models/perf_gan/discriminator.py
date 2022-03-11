import torch
import torch.nn as nn
from typing import List

from perf_gan.models.blocks.conv_blocks import ConvBlock
from perf_gan.models.blocks.linear_blocks import LinBlock


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
                      dilation=dilation,
                      norm=False,
                      dropout=0.4)
            for in_channels, out_channels, dilation in zip(
                conv_channels[:-1], conv_channels[1:], dilations)
        ])

        self.rnns = nn.ModuleList([
            nn.GRU(out_channels, out_channels, batch_first=True)
            for out_channels in conv_channels[1:]
        ])

        self.linears = nn.ModuleList([
            LinBlock(in_features=in_features, out_features=out_features)
            for in_features, out_features in zip(h_dims[:-1], h_dims[1:])
        ])
        self.mp = nn.MaxPool1d(4)

        self.__initialize_weights()

    def __initialize_weights(self) -> None:
        """Initialize weights of the discriminator (help training)
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pass forward

        Args:
            x (torch.Tensor): input contours of size (B, C, L)

        Returns:
            torch.Tensor: output tensor of size (B, 1, 1)
        """

        for conv, rnn in zip(self.conv, self.rnns):
            rnn.flatten_parameters()
            x = conv(x)
            x = x.permute(0, 2, 1)
            x, _ = rnn(x)
            x = x.permute(0, 2, 1)
            x = self.mp(x)

        x = nn.Flatten()(x)
        x = x.unsqueeze(1)

        for l in self.linears:
            x = l(x)

        return x
