import torch
import torch.nn as nn
from perf_gan.models.conv_blocks import ConvTblock, ConvBlock


class Discriminator(nn.Module):
    def __init__(self, img_c: int, hidden_c: int, factors=[1, 4, 8]) -> None:
        """Create discriminator tailored for MNIST dataset

        Args:
            img_c (int): number of channels of input image 
            hidden_c (int): number of channels of the hidden layer
            factors (list, optional): convolutional factors. Defaults to [1, 4, 8].
        """
        super(Discriminator, self).__init__()

        self.in_block = nn.Sequential(
            nn.Conv2d(img_c,
                      hidden_c * factors[0],
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(0.2),
        )
        self.down_blocks = nn.ModuleList([
            ConvBlock(hidden_c * factors[i],
                      hidden_c * factors[i + 1],
                      kernel_size=4,
                      stride=2,
                      padding=1) for i in range(len(factors) - 1)
        ])

        self.out_block = nn.Conv2d(hidden_c * factors[-1],
                                   1,
                                   kernel_size=2,
                                   stride=2,
                                   padding=0)  # size N x 1 x 1 x 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass for the discriminator

        Args:
            x (torch.Tensor): input image (fake or real)

        Returns:
            torch.Tensor: estimated probability of real image
        """
        x = self.in_block(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.out_block(x)
        return torch.sigmoid(x)