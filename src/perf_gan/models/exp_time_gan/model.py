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

        self.gen = Generator(self.channels, dropout=0.5)
        self.disc = Discriminator(self.disc_channels,
                                  self.disc_h_dims,
                                  dropout=0.5)

        self.reg = reg

        self.criteron = Hinge_loss()
        self.midi_loss = Midi_loss(f0_threshold=0.3, lo_threshold=2).cuda()

        self.f0_ratio = 1
        self.lo_ratio = 1

        self.val_idx = 0
        self.train_idx = 0
        self.automatic_optimization = False
        self.ddsp = None

        self.train_set = ContoursDataset(path="data/train_aug.pickle",
                                         list_transforms=list_transforms)

        self.test_set = ContoursDataset(path="data/test_c.pickle",
                                        list_transforms=list_transforms)

        self.save_hyperparameters()

    def forward(self, u_c, e_c):

        # reconstruction
        rec_c = self.ae(e_c)

        # generation
        gen_c = self.gen(u_c)

        return rec_c, gen_c

    def gen_step(self, u_c, e_c, g_c, mask):

        disc_e = self.disc(e_c).view(-1)
        disc_gen = self.disc(g_c).view(-1)

        gen_loss = self.criteron.gen_loss(disc_e, disc_gen)

        # decode gen / exp contours
        g_c = self.ae.decode(g_c)
        e_c = self.ae.decode(e_c)
        # Apply regularization

        u_f0, u_lo = u_c.split(1, 1)
        g_f0, g_lo = g_c.split(1, 1)

        # apply inverse transform to compare pitches (midi range) and loudness (loudness range)
        inv_u_f0, inv_u_lo = self.train_set.inverse_transform(u_f0, u_lo)
        inv_g_f0, inv_g_lo = self.train_set.inverse_transform(g_f0, g_lo)

        # add pitch loss
        f0_loss, lo_loss = self.midi_loss(inv_g_f0, inv_u_f0, inv_g_lo,
                                          inv_u_lo, mask)

        return gen_loss, f0_loss, lo_loss

    def disc_step(self, u_c, e_c, g_c):

        disc_e = self.disc(e_c).view(-1)
        disc_gen = self.disc(g_c.detach()).view(-1)

        disc_loss = self.criteron.disc_loss(disc_e, disc_gen)

        return disc_loss

    def training_step(self, batch, batch_idx):
        opt_ae, opt_gen, opt_disc = self.optimizers()

        u_f0, u_lo, e_f0, e_lo, onsets, offsets, mask = batch

        u_c = torch.cat([u_f0, u_lo], -2)
        e_c = torch.cat([e_f0, e_lo], -2)

        r_c, g_c = self(u_c, e_c)

        # train auto encoder
        rec_loss = torch.nn.functional.mse_loss(r_c, e_c)

        opt_ae.zero_grad()
        self.manual_backward(rec_loss)
        opt_ae.step()

        # train GAN
        # train Discriminator

        e_c = self.ae.encode(e_c)

        disc_loss = self.disc_step(u_c, e_c, g_c)
        self.disc.zero_grad()
        self.manual_backward(disc_loss)
        opt_disc.step()

        # train Generator
        gen_loss, f0_loss, lo_loss = self.gen_step(u_c, e_c, g_c, mask)

        e_c = self.ae.decode(e_c)
        g_c = self.ae.decode(g_c)

        # regularization : freeze decoder

        self.ae.decoder.requires_grad_(False)
        opt_gen.zero_grad()
        self.manual_backward(gen_loss + f0_loss + lo_loss)
        opt_gen.step()
        self.ae.decoder.requires_grad_(True)

        self.do_logs("train", gen_loss, disc_loss, rec_loss, f0_loss, lo_loss)

        self.train_idx += 1

        return {"u_c": u_c, "e_c": e_c, "r_c": r_c, "g_c": g_c}

    def validation_step(self, batch, batch_idx):

        u_f0, u_lo, e_f0, e_lo, onsets, offsets, mask = batch

        u_c = torch.cat([u_f0, u_lo], -2)
        e_c = torch.cat([e_f0, e_lo], -2)

        r_c, g_c = self(u_c, e_c)

        # test auto encoder
        rec_loss = torch.nn.functional.mse_loss(r_c, e_c)

        # test GAN

        e_c = self.ae.encode(e_c)

        disc_loss = self.disc_step(u_c, e_c, g_c)
        gen_loss, f0_loss, lo_loss = self.gen_step(u_c, e_c, g_c, mask)

        e_c = self.ae.decode(e_c)
        g_c = self.ae.decode(g_c)

        self.do_logs("val", gen_loss, disc_loss, rec_loss, f0_loss, lo_loss)

        self.val_idx += 1

        return {"u_c": u_c, "e_c": e_c, "r_c": r_c, "g_c": g_c}

    def do_logs(self, mode, gen_loss, disc_loss, rec_loss, f0_loss, lo_loss):
        self.log(f"{mode}/rec_loss", rec_loss)
        self.log(f"{mode}/f0_loss", f0_loss)
        self.log(f"{mode}/lo_loss", lo_loss)

        if mode == "train":
            self.logger.experiment.add_scalars(
                "abversarial",
                {
                    "gen": gen_loss,
                    "disc": disc_loss
                },
                global_step=self.train_idx,
            )

    def post_processing(self, outputs, c):
        last_c = outputs[-1][c]

        f0, lo = last_c[-1, ...].split(1, -2)

        # apply inverse transforms
        f0, lo = self.train_set.inverse_transform(f0, lo)

        # convert midi to hz / db
        f0 = self.__midi2hz(f0[0])
        f0 = f0.squeeze().cpu().detach()
        lo = lo.squeeze().cpu().detach()

        return f0, lo

    def training_epoch_end(self, outputs):

        u_f0, u_lo = self.post_processing(outputs, "u_c")
        e_f0, e_lo = self.post_processing(outputs, "e_c")
        r_f0, r_lo = self.post_processing(outputs, "r_c")
        g_f0, g_lo = self.post_processing(outputs, "g_c")

        # plot reconstruction

        plt.plot(e_f0, label="e_f0")
        plt.plot(r_f0, label="rec_f0")
        plt.legend()
        self.logger.experiment.add_figure("rec/f0", plt.gcf(), self.train_idx)

        plt.plot(e_lo, label="e_lo")
        plt.plot(r_lo, label="rec_lo")
        plt.legend()
        self.logger.experiment.add_figure("rec/lo", plt.gcf(), self.train_idx)

        #plot generation

        plt.plot(u_f0, label="u_f0")
        plt.plot(g_f0, label="gen_f0")
        plt.legend()
        self.logger.experiment.add_figure("gen/f0", plt.gcf(), self.train_idx)

        plt.plot(u_lo, label="u_lo")
        plt.plot(g_lo, label="gen_lo")
        plt.legend()
        self.logger.experiment.add_figure("gen/lo", plt.gcf(), self.train_idx)

    def __midi2hz(self, x):
        return torch.pow(2, (x - 69) / 12) * 440

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
        return DataLoader(dataset=self.train_set,
                          batch_size=128,
                          shuffle=True,
                          num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=128, num_workers=8)


if __name__ == "__main__":

    list_transforms = [(PitchTransform, {
        "feature_range": (-1, 1)
    }), (LoudnessTransform, {
        "feature_range": (-1, 1)
    })]

    n_sample = 1024
    channels = [2, 128, 256, 512]
    disc_channels = [512, 1024, 512, 256]
    div = 2**(len(channels) + len(channels) - 2)
    in_size = int(disc_channels[-1] * n_sample / div)

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
                         log_every_n_steps=5)

    trainer.fit(model)
