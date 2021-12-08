import torch

from perf_gan.utils.init_weights import initialize_weights
from perf_gan.models.generator import Generator
from perf_gan.models.discriminator import Discriminator


def test_gen_dims():
    N, z_dim, H, W = 8, 100, 1, 1
    x = torch.randn((N, z_dim, H, W))
    gen = Generator(z_dim, hidden_c=128, img_c=1, factors=[12, 4, 1])
    initialize_weights(gen)
    assert gen(x).shape == (N, 1, 28, 28)


def test_disc_dims():
    N, in_channels, H, W = 8, 1, 28, 28
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, hidden_c=8, factors=[1, 4, 12])
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)