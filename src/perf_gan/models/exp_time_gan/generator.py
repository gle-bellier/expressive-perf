import torch
import torch.nn as nn

from perf_gan.models.blocks.conv_blocks import ConvBlock


class GeneratorBlock(nn.Module):

    def __init__(self, in_c, out_c, dilation, dropout):
        super(GeneratorBlock, self).__init__()

        self.in_conv = ConvBlock(in_c, in_c, dilation, dropout)
        self.main = ConvBlock(in_c, out_c, dilation, dropout)
        self.res = ConvBlock(in_c, out_c, dilation, dropout)
        self.out_conv = ConvBlock(out_c, out_c, dilation, dropout)

        self.avg = nn.AvgPool1d(2)

    def forward(self, x):
        x = self.in_conv(x)

        main = self.main(x)
        res = self.res(x)
        main = self.avg(main)
        res = self.avg(res)
        x = res + self.out_conv(main)
        return x


class Generator(nn.Module):

    def __init__(self, channels, dropout=0):

        super(Generator, self).__init__()

        self.channels = channels
        self.encoder = nn.ModuleList([
            GeneratorBlock(in_c, out_c, dilation=1, dropout=dropout)
            for in_c, out_c in zip(channels[:-1], channels[1:])
        ])

    def forward(self, x):

        for block in self.encoder:
            x = block(x)
        return x
