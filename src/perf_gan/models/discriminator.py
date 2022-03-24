import torch
import torch.nn as nn
from typing import List

from perf_gan.models.blocks.conv_blocks import ConvBlock
from perf_gan.models.blocks.linear_blocks import LinBlock


class Discriminator(nn.Module):
    """Discriminator for performance contours modelling relying on a U-Net architecture
    """

    def __init__(self,
                 channels: List[int],
                 n_layers: List[int],
                 n_sample: int,
                 dropout=0.) -> None:

        super(Discriminator, self).__init__()

        n = int(n_sample / 4**(len(channels) - 1)) * channels[-1]

        ratio = torch.log2(torch.tensor(n))
        r = torch.linspace(0, ratio, n_layers)
        h_dims = torch.pow(2, r.to(int)).flip(dims=[0])

        self.rnns = nn.ModuleList(
            [nn.GRU(in_c, in_c, batch_first=True) for in_c in channels[:-1]])

        self.conv = nn.ModuleList([
            nn.Sequential(ConvBlock(in_c, out_c, pool=True, dropout=dropout),
                          ConvBlock(out_c, out_c, pool=True, dropout=dropout))
            for in_c, out_c in zip(channels[:-1], channels[1:])
        ])

        self.linears = nn.ModuleList([
            LinBlock(in_features, out_features)
            for in_features, out_features in zip(h_dims[:-1], h_dims[1:])
        ])

        self.__initialize_weights()

    def __initialize_weights(self) -> None:
        """Initialize weights of the discriminator (help training)
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pass forward

        Args:
            x (torch.Tensor): input contours of size (B, C, L)

        Returns:
            torch.Tensor: output tensor of size (B, 1, 1)
        """

        for conv, rnn in zip(self.conv, self.rnns):
            rnn.flatten_parameters()
            x = x.permute(0, 2, 1)
            x, _ = rnn(x)
            x = x.permute(0, 2, 1)

            x = conv(x)

        x = nn.Flatten()(x)
        x = x.unsqueeze(1)

        for l in self.linears:
            x = l(x)

        return x
