import torch
import torch.nn as nn
from perf_gan.models.conv_blocks import ConvBlock1D


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lr = nn.LeakyReLU()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.mp = nn.MaxPool1d(kernel_size=2)
        pass

    def forward(self, x):
        x = self.conv1(x)
        ctx = torch.clone(x)
        x = self.conv2(x)
        out = self.mp(x)
        return out, ctx