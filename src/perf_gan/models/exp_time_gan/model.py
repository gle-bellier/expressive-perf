import torch
import torch.nn as nn

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import matplotlib.pyplot as plt

from perf_gan.models.exp_time_gan.generator import Generator
from perf_gan.models.exp_time_gan.discriminator import Discriminator

from perf_gan.models.exp_time_gan.ae import AutoEncoder

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

        self.ae = AutoEncoder(channels, dropout=0.5)

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
        rec_c = self.ae(e_c)

        # generation
        gen_c = self.gen(u_c)

        return rec_c, gen_c

    def gen_step(self, u_c, e_c, gen_c, mask):

        disc_e = self.disc(e_c).view(-1)
        disc_gen = self.disc(gen_c).view(-1)

        gen_loss = self.criteron.gen_loss(disc_e, disc_gen)

        return gen_loss

    def disc_step(self, u_c, e_c, gen_c):

        disc_e = self.disc(e_c).view(-1)
        disc_gen = self.disc(gen_c.detach()).view(-1)

        disc_loss = self.criteron.disc_loss(disc_e, disc_gen)

        return disc_loss

    def training_step(self, batch, batch_idx):
        opt_ae, opt_gen, opt_disc = self.optimizers()

        u_f0, u_lo, e_f0, e_lo, onsets, offsets, mask = batch

        u_c = torch.cat([u_f0, u_lo], -2)
        e_c = torch.cat([e_f0, e_lo], -2)

        rec_c, gen_c = self(u_c, e_c)

        # train auto encoder
        rec_loss = torch.nn.functional.mse_loss(rec_c, e_c)

        opt_ae.zero_grad()
        self.manual_backward(rec_loss)
        opt_ae.step()

        # train GAN
        # train discriminator

        disc_loss = self.disc_step(u_c, e_c, gen_c)
        self.disc.zero_grad()
        self.manual_backward(disc_loss)
        opt_disc.step()

        # train generator
        gen_loss = self.gen_step(u_c, e_c, gen_c, mask)

        opt_gen.zero_grad()
        self.manual_backward(gen_loss)
        opt_gen.step()

        # decode gen contours
        gen_c = self.ae.decode(gen_c)

        self.log("train/rec_loss", rec_loss)
        self.logger.experiment.add_scalars(
            "train/abversarial",
            {
                "gen": gen_loss,
                "disc": disc_loss
            },
            global_step=self.train_idx,
        )
        self.train_idx += 1

        return {"u_c": u_c, "e_c": e_c, "rec_c": rec_c, "gen_c": gen_c}

    def training_epoch_end(self, outputs):

        last_u_c = outputs[-1]["u_c"]
        last_e_c = outputs[-1]["e_c"]
        last_rec_c = outputs[-1]["rec_c"]
        last_gen_c = outputs[-1]["gen_c"]

        # plot last reconstruction
        u_f0, u_lo = last_u_c[-1, ...].split(1, -2)
        e_f0, e_lo = last_e_c[-1, ...].split(1, -2)
        rec_f0, rec_lo = last_rec_c[-1, ...].split(1, -2)
        gen_f0, gen_lo = last_gen_c[-1, ...].split(1, -2)

        # plot reconstruction

        plt.plot(e_f0.squeeze().cpu().detach(), label="e_f0")
        plt.plot(rec_f0.squeeze().cpu().detach(), label="rec_f0")
        plt.legend()
        self.logger.experiment.add_figure("rec/f0", plt.gcf(), self.train_idx)

        plt.plot(e_lo.squeeze().cpu().detach(), label="e_lo")
        plt.plot(rec_lo.squeeze().cpu().detach(), label="rec_lo")
        plt.legend()
        self.logger.experiment.add_figure("rec/lo", plt.gcf(), self.train_idx)

        #plot generation

        plt.plot(u_f0.squeeze().cpu().detach(), label="u_f0")
        plt.plot(gen_f0.squeeze().cpu().detach(), label="gen_f0")
        plt.legend()
        self.logger.experiment.add_figure("gen/f0", plt.gcf(), self.train_idx)

        plt.plot(u_lo.squeeze().cpu().detach(), label="u_lo")
        plt.plot(gen_lo.squeeze().cpu().detach(), label="gen_lo")
        plt.legend()
        self.logger.experiment.add_figure("gen/lo", plt.gcf(), self.train_idx)

    def configure_optimizers(self):
        """Configure both generator and discriminator optimizers

        Returns:
            Tuple(list): (list of optimizers, empty list) 
        """

        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_ae = torch.optim.Adam(self.ae.parameters(), lr=lr, betas=(b1, b2))

        opt_g = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(b1, b2))

        return opt_ae, opt_g, opt_d

    def train_dataloader(self):
        self.train_set = ContoursDataset(path="data/dataset_aug.pickle",
                                         list_transforms=self.list_transforms)
        return DataLoader(dataset=self.train_set,
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
    channels = [2, 128, 256, 512]
    disc_channels = [2, 16, 128, 512]
    div = 4**(len(channels) - 1)
    in_size = int(channels[-1] * n_sample / div)

    model = ExpTimeGAN(channels=channels,
                       disc_channels=disc_channels,
                       disc_h_dims=[in_size, 1024, 512, 64, 16, 1],
                       reg=True,
                       list_transforms=list_transforms,
                       lr=1e-3,
                       b1=0.5,
                       b2=0.999)

    #model.ddsp = torch.jit.load("ddsp_violin.ts").eval()

    tb_logger = pl_loggers.TensorBoardLogger('runs/')
    trainer = pl.Trainer(gpus=1,
                         max_epochs=10000,
                         logger=tb_logger,
                         log_every_n_steps=10)

    trainer.fit(model)
