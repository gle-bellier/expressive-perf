import torch
import torch.nn as nn
from typing import List

from perf_gan.models.blocks.conv_blocks import ConvBlock
from perf_gan.models.blocks.linear_blocks import LinBlock


class Discriminator(nn.Module):
    def __init__(self, channels, h_dims, dropout=0):

        super(Discriminator, self).__init__()

        self.conv = nn.ModuleList([
            ConvBlock(
                in_channels=in_c,
                out_channels=out_c,
                dilation=1,
                norm=False,
                dropout=dropout,
            ) for in_c, out_c in zip(channels[:-1], channels[1:])
        ])

        self.rnns = nn.ModuleList(
            [nn.GRU(out_c, out_c, batch_first=True) for out_c in channels[1:]])

        self.linears = nn.ModuleList([
            LinBlock(in_features, out_features)
            for in_features, out_features in zip(h_dims[:-1], h_dims[1:])
        ])
        self.avg = nn.AvgPool1d(4)

    def forward(self, x):

        for conv, rnn in zip(self.conv, self.rnns):
            rnn.flatten_parameters()
            x = conv(x)
            x = x.permute(0, 2, 1)
            x, _ = rnn(x)
            x = x.permute(0, 2, 1)
            x = self.avg(x)

        x = nn.Flatten()(x)
        x = x.unsqueeze(1)

        for l in self.linears:
            x = l(x)

        return x
