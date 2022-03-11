import torch
import torch.nn as nn

from perf_gan.models.blocks.conv_blocks import ConvBlock, ConvTransposeBlock


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, dilation, dropout, bidirectional=True):
        super(DecoderBlock, self).__init__()

        self.D = 1 + bidirectional
        self.in_conv = ConvTransposeBlock(in_c, in_c, dilation, dropout)
        self.out_conv = ConvTransposeBlock(in_c * self.D, out_c, dilation,
                                           dropout)

        self.gru = nn.GRU(in_c,
                          in_c,
                          batch_first=True,
                          bidirectional=bidirectional)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = self.in_conv(x)
        print("out in conv : ", x.shape)
        pass

        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x.permute(0, 2, 1)

        print("out gru conv : ", x.shape)

        x = self.up(x)

        print("out upsampling ", x.shape)
        x = self.out_conv(x)

        print("out out conv : ", x.shape)
        return x


class Decoder(nn.Module):
    def __init__(self, channels, dropout=0):

        super(Decoder, self).__init__()

        self.channels = channels
        self.decoder = nn.ModuleList([
            DecoderBlock(in_c,
                         out_c,
                         dilation=1,
                         dropout=dropout,
                         bidirectional=True)
            for in_c, out_c in zip(channels[:-1], channels[1:])
        ])

    def forward(self, x):

        print(x.shape)

        for block in self.decoder:
            x = block(x)
            print(x.shape)

        return x
