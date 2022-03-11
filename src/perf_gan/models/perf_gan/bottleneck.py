import torch
import torch.nn as nn


class Bottleneck(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Create bottleneck for U-Net architecture

        Args:
            in_channels (int): input number of channels 
            out_channels (int): ouput number of channels
        """
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=4,
                              padding=1,
                              stride=2)

        self.gru = nn.GRU(out_channels, out_channels, batch_first=True)
        self.lr = nn.LeakyReLU(.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass

        Args:
            x (torch.Tensor): input contours

        Returns:
            torch.Tensor: output contours"""
        x = self.conv(x)

        # apply rnn
        self.gru.flatten_parameters()
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x.permute(0, 2, 1)

        out = self.lr(x)

        return out