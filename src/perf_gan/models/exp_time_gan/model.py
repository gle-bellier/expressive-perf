import torch
import torch.nn as nn

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from perf_gan.models.exp_time_gan.generator import Generator
from perf_gan.models.exp_time_gan.discriminator import Discriminator

from perf_gan.models.exp_time_gan.encoder import Encoder
from perf_gan.models.exp_time_gan.decoder import Decoder

from perf_gan.data.contours_dataset import ContoursDataset
from perf_gan.data.preprocess import PitchTransform, LoudnessTransform

from perf_gan.losses.hinge_loss import Hinge_loss
from perf_gan.losses.midi_loss import Midi_loss

import warnings

warnings.filterwarnings('ignore')


class ExpTimeGAN(pl.LightningModule):
    def __init__(self, channels, disc_channels, disc_h_dims):
        super(ExpTimeGAN, self).__init__()

        self.channels = channels
        self.disc_channels = disc_channels
        self.disc_h_dims = disc_h_dims

        self.encoder = Encoder(self.channels, dropout=0)
        self.decoder = Decoder(self.channels[::-1], dropout=0)

        self.gen = Generator(self.channels, dropout=0)
        self.disc = Discriminator(self.disc_channels,
                                  self.disc_h_dims,
                                  dropout=0)

        self.save_hyperparameters()

        self.midi_loss = Midi_loss(f0_threshold=0.3, lo_threshold=2).cuda()

    def forward(self, u_c, e_c):

        # reconstruction

        h = self.encoder(e_c)
        recons = self.decoder(h)

        gen_c = self.gen(u_c)

        return recons, gen_c


if __name__ == "__main__":

    n_sample = 1024

    channels = [2, 16, 128, 512]
    disc_channels = [2, 16, 128, 512]

    channels = [2, 16, 128, 512]
    div = 4**(len(channels) - 1)
    in_size = int(channels[-1] * n_sample / div)

    model = ExpTimeGAN(channels=[2, 16, 128, 512],
                       disc_channels=[2, 16, 128, 512],
                       disc_h_dims=[in_size, 1024, 512, 64, 16, 1])

    u_c = e_c = torch.randn(32, 2, 1024)
    model(u_c, e_c)