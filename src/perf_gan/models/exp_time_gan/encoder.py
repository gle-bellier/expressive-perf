import torch
import torch.nn as nn

from perf_gan.models.blocks.conv_blocks import ConvBlock


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, dilation, dropout, bidirectional=True):
        super(EncoderBlock, self).__init__()

        self.D = 1 + bidirectional
        self.in_conv = ConvBlock(in_c, in_c, dilation, dropout)
        self.out_conv = ConvBlock(in_c * self.D, out_c, dilation, dropout)

        self.gru = nn.GRU(in_c,
                          in_c,
                          batch_first=True,
                          bidirectional=bidirectional)
        self.avg = nn.AvgPool1d(2)

    def forward(self, x):
        x = self.in_conv(x)

        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x.permute(0, 2, 1)

        x = self.avg(x)
        x = self.out_conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, channels, dropout=0):

        super(Encoder, self).__init__()

        self.channels = channels
        self.encoder = nn.ModuleList([
            EncoderBlock(in_c,
                         out_c,
                         dilation=1,
                         dropout=dropout,
                         bidirectional=True)
            for in_c, out_c in zip(channels[:-1], channels[1:])
        ])

    def forward(self, x):

        for block in self.encoder:
            x = block(x)
            print(x.shape)

        return x
