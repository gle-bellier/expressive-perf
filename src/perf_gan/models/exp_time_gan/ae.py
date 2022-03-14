import torch
import torch.nn as nn

from perf_gan.models.exp_time_gan.encoder import Encoder
from perf_gan.models.exp_time_gan.decoder import Decoder


class AutoEncoder(nn.Module):

    def __init__(self, channels, dropout):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(channels, dropout=dropout)
        self.decoder = Decoder(channels[::-1], dropout=dropout)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return torch.tanh(x)
