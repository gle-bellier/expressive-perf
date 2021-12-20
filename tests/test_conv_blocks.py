import pytest
import torch

from perf_gan.models.conv_blocks import ConvBlock
from perf_gan.models.u_block import UBlock
from perf_gan.models.d_block import DBlock


def test_dims_conv1d():
    batch_size = 10
    in_channels = 32
    length = 100
    out_channels = 64

    in_c = torch.randn((
        batch_size,
        in_channels,
        length,
    ))

    assert ConvBlock(in_channels, out_channels,
                     dilation=3)(in_c).shape == (batch_size, out_channels,
                                                 length)


def test_upsampling_block():

    batch_size = 10
    in_channels = 32
    length = 100
    out_channels = 64
    dilation = 3

    x = torch.randn((
        batch_size,
        in_channels,
        length,
    ))

    ublock = UBlock(in_channels, out_channels, dilation=dilation)
    assert ublock(x).shape == (batch_size, out_channels, length * 2)


def test_downsampling_block():

    batch_size = 10
    in_channels = 32
    length = 100
    out_channels = 64
    dilation = 3

    x = torch.randn((
        batch_size,
        in_channels,
        length,
    ))

    ublock = DBlock(in_channels, out_channels, dilation=dilation)
    assert ublock(x).shape == (batch_size, out_channels, length // 2)
