import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pool=False,
                 norm=True,
                 dropout=0.) -> None:

        super().__init__()

        stride = 2 if pool else 1
        self.norm = norm
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              dilation=1,
                              padding=self.__get_padding(3, 1, 1),
                              stride=stride)

        self.lr = nn.LeakyReLU(.2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dp = nn.Dropout(dropout)

    def __get_padding(self, kernel_size, stride: int, dilation: int) -> int:
        """Return size of the padding needed

        Args:
            kernel_size ([type]): kernel size of the convolutional layer
            stride (int): stride of the convolutional layer
            dilation (int): dilation of the convolutional layer

        Returns:
            int: padding
        """
        full_kernel = (kernel_size - 1) * dilation + 1
        return full_kernel // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass 

        Args:
            x (torch.Tensor): input contours

        Returns:
            torch.Tensor: output contours
        
        """
        x = self.dp(x)
        x = self.conv(x)

        if self.norm:
            x = self.bn(x)
        out = self.lr(x)

        return out


class ConvTransposeBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 upsample=False,
                 norm=True,
                 dropout=0.) -> None:
        """Create 1D Convolutional block composed of a convolutional layer
        followed by batch normalization and leaky ReLU.

        Args:
            in_channels (int): input number of channels
            out_channels (int): output number of channels
            dilation (int): dilation of the convolutional layer
            norm (bool, optional): process batchnorm. Defaults to False.
            dropout (float, optional): dropout probability. Defaults to 0.
        """

        super().__init__()

        stride = 2 if upsample else 1
        self.norm = norm
        self.conv = nn.ConvTranspose1d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       dilation=1,
                                       padding=self.__get_padding(3, 1, 1),
                                       stride=stride,
                                       output_padding=stride - 1)

        self.lr = nn.LeakyReLU(.2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dp = nn.Dropout(dropout)

    def __get_padding(self, kernel_size, stride: int, dilation: int) -> int:
        """Return size of the padding needed

        Args:
            kernel_size ([type]): kernel size of the convolutional layer
            stride (int): stride of the convolutional layer
            dilation (int): dilation of the convolutional layer

        Returns:
            int: padding
        """
        full_kernel = (kernel_size - 1) * dilation + 1
        return full_kernel // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass 

        Args:
            x (torch.Tensor): input contours

        Returns:
            torch.Tensor: output contours
        
        """
        x = self.dp(x)
        x = self.conv(x)

        if self.norm:
            x = self.bn(x)

        out = self.lr(x)

        return out
