import torch
import torch.nn as nn


class ConvTblock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, padding: int) -> None:
        """Transpose Convolution block with convolution followed by LeakyRelu

        Args:
            in_channels (int): in_channels of convolution transpose
            out_channels (int): out_channels of convolution transpose
            kernel_size (int): kernel_size of convolution transpose
            stride (int): stride of convolution transpose
            padding (int): padding of convolution transpose
        """
        super(ConvTblock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels,
                               out_channels,
                               kernel_size,
                               stride,
                               padding,
                               bias=False), nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass for a given sample 

        Args:
            x (torch.Tensor): sample from the dataset

        Returns:
            torch.Tensor: result of the forward pass
        """
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, padding: int) -> None:
        """Convolutional block with convolution followed by LeakyRelu

        Args:
            in_channels (int): in_channels of convolution
            out_channels (int): out_channels of convolution
            kernel_size (int): kernel_size of convolution
            stride (int): stride of convolution
            padding (int): padding of convolution
        """
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding,
                      bias=False), nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the forward pass for a given sample 

        Args:
            x (torch.Tensor): sample from the dataset

        Returns:
            torch.Tensor: result of the forward pass
        """
        return self.block(x)