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
    def __init__(self, channels, disc_channels, disc_h_dims, reg,
                 list_transforms, lr, b1, b2):
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

        self.reg = reg

        self.criteron = Hinge_loss()
        self.midi_loss = Midi_loss(f0_threshold=0.3, lo_threshold=2).cuda()

        self.f0_ratio = 1
        self.lo_ratio = 1

        self.val_idx = 0
        self.train_idx = 0
        self.automatic_optimization = False
        self.ddsp = None

        self.train_set = None
        self.test_set = None
        self.list_transforms = list_transforms

        self.save_hyperparameters()

    def forward(self, u_c, e_c):

        # reconstruction

        h = self.encoder(e_c)
        recons = self.decoder(h)

        gen_c = self.gen(u_c)

        return recons, gen_c

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

        u_f0, u_lo, e_f0, e_lo, onsets, offsets, mask = batch

        u_contours = torch.cat([u_f0, u_lo], -2)
        e_contours = torch.cat([e_f0, e_lo], -2)

    def configure_optimizers(self):
        """Configure both generator and discriminator optimizers

        Returns:
            Tuple(list): (list of optimizers, empty list) 
        """

        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_encoder = torch.optim.Adam(self.gen.parameters(),
                                       lr=lr,
                                       betas=(b1, b2))
        opt_decoder = torch.optim.Adam(self.disc.parameters(),
                                       lr=lr,
                                       betas=(b1, b2))

        opt_g = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(b1, b2))

        return opt_encoder, opt_decoder, opt_g, opt_d

    def train_dataloader(self):
        self.train_set = ContoursDataset(path="data/dataset_aug.pickle",
                                         list_transforms=self.list_transforms)
        train_dataloader = DataLoader(dataset=self.train_set,
                                      batch_size=64,
                                      shuffle=True,
                                      num_workers=8)

    def val_dataloader(self):
        self.test_set = ContoursDataset(path="data/dataset_aug.pickle",
                                        list_transforms=self.list_transforms)
        return DataLoader(self.test_set, batch_size=64, num_workers=8)


if __name__ == "__main__":

    list_transforms = [(PitchTransform, {
        "feature_range": (-1, 1)
    }), (LoudnessTransform, {
        "feature_range": (-1, 1)
    })]

    n_sample = 1024
    channels = [2, 16, 128, 512]
    disc_channels = [2, 16, 128, 512]
    div = 4**(len(channels) - 1)
    in_size = int(channels[-1] * n_sample / div)

    model = ExpTimeGAN(channels=[2, 16, 128, 512],
                       disc_channels=[2, 16, 128, 512],
                       disc_h_dims=[in_size, 1024, 512, 64, 16, 1],
                       reg=True,
                       list_transforms=list_transforms,
                       lr=1e-3,
                       b1=0.5,
                       b2=0.999)

    #model.ddsp = torch.jit.load("ddsp_violin.ts").eval()

    tb_logger = pl_loggers.TensorBoardLogger('runs/')
    trainer = pl.Trainer(gpus=0,
                         max_epochs=10000,
                         logger=tb_logger,
                         log_every_n_steps=10)

    trainer.fit(model)
