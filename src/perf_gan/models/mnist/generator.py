import torch
import torch.nn as nn
from perf_gan.models.mnist.conv_blocks_2d import ConvTblock, ConvBlock


class Generator(nn.Module):
    def __init__(self,
                 z_c: torch.Tensor,
                 hidden_c: int,
                 img_c: int,
                 factors=[8, 4, 1]) -> None:
        """Create generator for GAN tailored for MNIST dataset

        Args:
            z_c (torch.Tensor): input noise
            hidden_c (int): hidden channels
            img_c (int): image channels
            factors (list, optional): list of convolutional factors. Defaults to [8, 4, 1].
        """
        super(Generator, self).__init__()

        self.in_block = ConvTblock(z_c, hidden_c * factors[0], 4, 1, 0)
        self.up_blocks = nn.ModuleList([
            ConvTblock(hidden_c * factors[i], hidden_c * factors[i + 1], 4, 2,
                       1) for i in range(len(factors) - 1)
        ])
        self.out_block = nn.Sequential(
            nn.ConvTranspose2d(hidden_c * factors[-1], img_c, 4, 2, 3),
            nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass for the generator

        Args:
            x (torch.Tensor): input noise

        Returns:
            torch.Tensor: generated image
        """

        x = self.in_block(x)
        for block in self.up_blocks:
            x = block(x)

        x = self.out_block(x)

        return x