import pytest
import torch

from perf_gan.models.conv_blocks_1d import ConvBlock1D


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

    assert ConvBlock1D(in_channels, out_channels,
                       dilation=3)(in_c).shape == (batch_size, out_channels,
                                                   length)
