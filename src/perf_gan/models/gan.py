import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import List, Tuple

from perf_gan.models.generator import Generator
from perf_gan.models.discriminator import Discriminator

from perf_gan.data.dataset import GANDataset
from perf_gan.data.preprocess import PitchTransform, LoudnessTransform

from perf_gan.losses.lsgan_loss import LSGAN_loss
from perf_gan.losses.hinge_loss import Hinge_loss
from perf_gan.losses.midi_loss import Midi_loss


class PerfGAN(pl.LightningModule):
    def __init__(self, g_down_channels: List[int], g_up_channels: List[int],
                 g_down_dilations: List[int], g_up_dilations: List[int],
                 d_conv_channels: List[int], d_dilations: List[int],
                 d_h_dims: List[int], criteron: float, lr: float, b1: int,
                 b2: int):
        """[summary]

        Args:
            g_down_channels (List[int]): generator list of downsampling channels
            g_up_channels (List[int]): generator list of upsampling channels
            g_down_dilations (List[int]): generator list of down blocks dilations
            g_up_dilations (List[int]): generator list of up blocks dilations
            d_conv_channels (List[int]): discriminator list of convolutional channels
            d_dilations (List[int]): discriminator list of dilations
            d_h_dims (List[int]): discriminator list of hidden dimensions
            criteron (float): criteron for both generator and discriminator
            lr (float): learning rate
            b1 (int): b1 factor 
            b2 (int): b2 factor

        """
        super(PerfGAN, self).__init__()

        self.save_hyperparameters()

        self.gen = Generator(down_channels=g_down_channels,
                             up_channels=g_up_channels,
                             down_dilations=g_down_dilations,
                             up_dilations=g_up_dilations)

        self.disc = Discriminator(conv_channels=d_conv_channels,
                                  dilations=d_dilations,
                                  h_dims=d_h_dims)

        self.criteron = criteron
        self.dataset = None
        self.midi_loss = Midi_loss().cuda()

        self.val_idx = 0

        self.pitch_ratio = 1
        self.lo_ratio = 1

        self.automatic_optimization = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute generator pass forward with unexpressive contours
        (MIDI)

        Args:
            x (torch.Tensor): unexpressive contours (B, C, L)

        Returns:
            torch.Tensor: generated expressive contours (B, C, L)
        """
        return self.gen(x)

    def gen_step(self, u_contours, e_contours, gen_contours):

        disc_e = self.disc(e_contours).view(-1)
        disc_gu = self.disc(gen_contours).view(-1)

        gen_loss = self.criteron.gen_loss(disc_e, disc_gu)

        # apply inverse transform to compare pitches (midi range) and loudness (midi range)
        # inv_u_f0, inv_u_lo = self.dataset.inverse_transform(u_contours).split(
        #     1, 1)
        # inv_gen_f0, inv_gen_lo = self.dataset.inverse_transform(
        #     gen_contours).split(1, 1)

        # add pitch loss

        # pitch_loss, lo_loss = self.midi_loss(inv_gen_f0,
        #                                      inv_u_f0,
        #                                      inv_gen_lo,
        #                                      inv_u_lo,
        #                                      onsets,
        #                                      offsets,
        #                                      types=["mean", "mean"])

        pitch_loss = lo_loss = 0

        return gen_loss, pitch_loss, lo_loss

    def disc_step(self, u_contours, e_contours, gen_contours):

        # discriminate
        disc_e = self.disc(e_contours).view(-1)
        disc_u = self.disc(gen_contours.detach()).view(-1)

        disc_loss = self.criteron.disc_loss(disc_e, disc_u)

        return disc_loss

    def training_step(self, batch: List[torch.Tensor], batch_idx: int,
                      optimizer_idx: int) -> OrderedDict:
        """Compute a training step for generator or discriminator 
        (according to optimizer index)

        Args:
            batch (List[torch.Tensor]): batch composed of (u_contours, e_contours, onsets, offsets)
            batch_idx (int): batch index
            optimizer_idx (int): optimizer index (0 for generator, 1 for discriminator)

        Returns:
            OrderedDict: dict {loss, progress_bar, log}
        """

        g_opt, d_opt = self.optimizers()

        u_contours, e_contours, onsets, offsets = batch
        # generate new contours
        gen_contours = self.gen(u_contours)

        # train discriminator
        disc_loss = self.disc_step(u_contours, e_contours, gen_contours)

        self.disc.zero_grad()
        self.manual_backward(disc_loss)
        d_opt.step()

        # train generator

        gen_loss, pitch_loss, lo_loss = self.gen_step(u_contours, e_contours,
                                                      gen_contours)

        g_opt.zero_grad()
        self.manual_backward(gen_loss)
        g_opt.step()

        # self.log("gen_pitch_loss", pitch_loss)
        # self.log("gen_lo_loss", lo_loss)
        self.log_dict({"g_loss": gen_loss, "d_loss": disc_loss}, prog_bar=True)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Compute validation step (do some logging)

        Args:
            batch (torch.Tensor): batch composed of (u_contours, e_contours, onsets, offsets)
            batch_idx (int): batch index
        """
        self.val_idx += 1
        if self.val_idx % 100 == 0:

            u_contours, e_contours, _, _ = batch
            fake_contours = self.gen(u_contours)

            u_f0, u_lo = u_contours[0].split(1, 0)
            e_f0, e_lo = e_contours[0].split(1, 0)
            g_f0, g_lo = fake_contours[0].split(1, 0)

            plt.plot(u_f0.squeeze().cpu().detach(), label="u_f0")
            plt.plot(e_f0.squeeze().cpu().detach(), label="e_f0")
            plt.plot(g_f0.squeeze().cpu().detach(), label="g_f0")
            plt.legend()
            self.logger.experiment.add_figure("pitch", plt.gcf(), self.val_idx)

            plt.plot(u_lo.squeeze().cpu().detach(), label="u_lo")
            plt.plot(e_lo.squeeze().cpu().detach(), label="e_lo")
            plt.plot(g_lo.squeeze().cpu().detach(), label="g_lo")
            plt.legend()
            self.logger.experiment.add_figure("lo", plt.gcf(), self.val_idx)

    def configure_optimizers(self) -> Tuple:
        """Configure both generator and discriminator optimizers

        Returns:
            Tuple(list): (list of optimizers, empty list) 
        """

        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(b1, b2))
        #opt_d = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.SGD(self.disc.parameters(), lr=lr, momentum=.5)

        return opt_g, opt_d


if __name__ == "__main__":
    # get dataset
    list_transforms = [(PitchTransform, {
        "feature_range": (-1, 1)
    }), (LoudnessTransform, {
        "feature_range": (-1, 1)
    })]
    train_set = GANDataset(path="data/train.pickle",
                           n_sample=1024,
                           list_transforms=list_transforms)
    train_dataloader = DataLoader(dataset=train_set,
                                  batch_size=16,
                                  shuffle=True,
                                  num_workers=8)
    test_set = GANDataset(path="data/test.pickle",
                          n_sample=1024,
                          list_transforms=list_transforms)
    test_dataloader = DataLoader(dataset=test_set,
                                 batch_size=16,
                                 shuffle=True,
                                 num_workers=8)

    criteron = Hinge_loss()
    # init model
    model = PerfGAN(g_down_channels=[2, 4, 8, 16],
                    g_up_channels=[32, 16, 8, 4, 2],
                    g_down_dilations=[1, 1, 3, 3],
                    g_up_dilations=[3, 3, 1, 1, 1],
                    d_conv_channels=[2, 32, 16, 8, 1],
                    d_dilations=[1, 1, 1, 3],
                    d_h_dims=[1024, 64, 32, 1],
                    criteron=criteron,
                    lr=1e-4,
                    b1=0.5,
                    b2=0.999)

    model.dataset = train_set

    tb_logger = pl_loggers.TensorBoardLogger('runs/')
    trainer = pl.Trainer(gpus=1, max_epochs=1000, logger=tb_logger)

    trainer.fit(model, train_dataloader, test_dataloader)
