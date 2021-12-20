from typing import List
import pytest
import torch

from perf_gan.models.generator import Generator


@pytest.mark.parametrize(
    "down_channels, up_channels, down_dilations, up_dilations",
    [
        ([2, 4, 8, 16], [32, 16, 8, 4, 2], [5, 5, 3, 3], [3, 3, 5, 7, 7]),
    ],
)
def test_generator_shape(down_channels: List[int], up_channels: List[int],
                         down_dilations: List[int], up_dilations: List[int]):
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
