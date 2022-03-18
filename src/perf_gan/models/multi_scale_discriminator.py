import torch
import torch.nn as nn
from typing import List

from perf_gan.models.blocks.conv_blocks import ConvBlock
from perf_gan.models.blocks.linear_blocks import LinBlock


class DiscriminatorBlock(nn.Module):
    def __init__(self, channels, n_sample, h_sizes):
        super(DiscriminatorBlock, self).__init__()

        self.pre = ConvBlock(channels[0], channels[1], 1)
        self.dwn = nn.AvgPool1d(kernel_size=4**4, stride=4**4)

        self.convs = nn.ModuleList([
            ConvBlock(in_c, out_c, 1)
            for in_c, out_c in zip(channels[1:-1], channels[2:])
        ])
        self.flatten = nn.Flatten()

        in_size = int((channels[-1] * n_sample) / (4**4))
        f_sizes = [in_size] + h_sizes

        print("n sample : ", n_sample)
        print("last channels : ", channels[-1])
        print("in sizes : ", in_size)

        self.mlp = nn.ModuleList([
            LinBlock(in_f, out_f)
            for in_f, out_f in zip(f_sizes[:-1], f_sizes[1:])
        ])

    def forward(self, x):

        x = self.pre(x)
        x = self.dwn(x)

        for conv in self.convs:
            x = conv(x)

        # flatten before mlp
        x = self.flatten(x)

        for layer in self.mlp:
            x = layer(x)

        return x


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, down_factors, channels, n_sample, h_sizes):
        super(MultiScaleDiscriminator, self).__init__()

        self.discs = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool1d(kernel_size=down_factor, stride=down_factor),
                DiscriminatorBlock(channels, n_sample // down_factor, h_sizes))
            for down_factor in down_factors
        ])

    def forward(self, x):

        results = []
        for disc in self.discs:
            results += [disc(x)]

        return results
