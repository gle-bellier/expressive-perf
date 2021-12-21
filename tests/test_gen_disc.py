from typing import List
import pytest
import torch

from perf_gan.models.generator import Generator
from perf_gan.models.discriminator import Discriminator


@pytest.mark.parametrize(
    "down_channels, up_channels, down_dilations, up_dilations",
    [
        ([2, 4, 8, 16], [32, 16, 8, 4, 2], [5, 5, 3], [3, 3, 5, 7]),
        ([2, 4, 8], [16, 8, 4, 2], [5, 3], [3, 5, 7]),
    ],
)
def test_generator_shape(down_channels: List[int], up_channels: List[int],
                         down_dilations: List[int],
                         up_dilations: List[int]) -> None:
    """Test generator output shape and pass forward

    Args:
        down_channels (List[int]): list of downsampling channels
        up_channels (List[int]): list of upsampling channels
        down_dilations (List[int]): list of dilation factors for downsampling blocks
        up_dilations (List[int]): list of dilation factors for upsampling blocks
    """
    x = torch.randn(16, 2, 256)
    gen = Generator(down_channels, up_channels, down_dilations, up_dilations)
    out = gen(x)
    assert x.shape == out.shape


@pytest.mark.parametrize(
    "conv_channels, dilations, h_dims",
    [([2, 4, 8, 1], [3, 5, 3], [256, 32, 4, 1]),
     ([2, 4, 1], [3, 3], [256, 32, 8, 1])],
)
def test_discriminator_shape(conv_channels: List[int], dilations: List[int],
                             h_dims: List[int]) -> None:
    x = torch.randn(16, 2, 256)
    disc = Discriminator(conv_channels, dilations, h_dims)
    assert disc(x).shape == (x.shape[0], 1, 1)


@pytest.mark.parametrize(
    "gen_params, disc_params",
    [(([2, 4, 8, 16], [32, 16, 8, 4, 2], [5, 5, 3], [3, 3, 5, 7]),
      ([2, 4, 8, 1], [3, 5, 3], [256, 32, 4, 1])),
     (([2, 4, 8], [16, 8, 4, 2], [5, 3], [3, 5, 7]),
      ([2, 4, 1], [3, 3], [256, 32, 8, 1]))],
)
def test_gen_disc_shape(gen_params: tuple, disc_params: tuple) -> None:
    """Test the whole pass forward (generator -> discriminator)

    Args:
        gen_params (tuple): generator parameters: down_channels, up_channels, down_dilations, up_dilations
        disc_params (tuple): discriminator parameters: conv_channels, dilations, h_dims
    """
    x = torch.randn(16, 2, 256)
    gen = Generator(*gen_params)
    disc = Discriminator(*disc_params)

    x = gen(x)
    assert disc(x).shape == (x.shape[0], 1, 1)