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
from perf_gan.models.melgan_disc import Discriminator

from perf_gan.data.contours_dataset import ContoursDataset
from perf_gan.data.preprocess import PitchTransform, LoudnessTransform
from perf_gan.utils.post_processing import logging, c2wav

from perf_gan.losses.lsgan_loss import LSGAN_loss
from perf_gan.losses.hinge_loss import Hinge_loss
from perf_gan.losses.midi_loss import Midi_loss

import warnings

warnings.filterwarnings('ignore')


class PerfGAN(pl.LightningModule):

    def __init__(self, g_down_channels: List[int], g_up_channels: List[int],
                 g_down_dilations: List[int], g_up_dilations: List[int],
                 d_conv_channels: List[int], d_dilations: List[int],
                 d_h_dims: List[int], criteron: float, regularization: bool,
                 lr: float, b1: int, b2: int):
        """[summary]

        Args:
            mode (str): equals to "dev" for the model generating pitch and loudness deviation or "contours" for full contours generation
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

        self.disc = Discriminator(num_D=3,
                                  ndf=16,
                                  n_layers=4,
                                  downsampling_factor=4)

        self.criteron = criteron
        self.reg = regularization
        self.midi_loss = Midi_loss(f0_threshold=0.3, lo_threshold=2).cuda()

        self.train_set = ContoursDataset(path="data/train_aug.pickle",
                                         list_transforms=list_transforms)

        self.test_set = ContoursDataset(path="data/test_c.pickle",
                                        list_transforms=list_transforms)
        self.inv_transform = self.train_set.inverse_transform

        self.val_idx = 0
        self.train_idx = 0

        self.pitch_ratio = 1
        self.lo_ratio = 1

        self.automatic_optimization = False

        self.ddsp = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute generator pass forward with unexpressive contours
        (MIDI)

        Args:
            x (torch.Tensor): unexpressive contours (B, C, L)

        Returns:
            torch.Tensor: generated expressive contours (B, C, L)
        """

        # generate deviations contours

        return self.gen(x) + x

    def gen_step(
            self, u_c: torch.Tensor, e_c: torch.Tensor, g_c: torch.Tensor,
            mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute generator loss according to criteron.
        Expressive contours are considered the real data. In case of pitch and loudness 
        regularization this function outputs non null pitch and loudness proper losses

        Args:
            u_c (torch.Tensor): unexpressive contours
            e_c (torch.Tensor): expressive contours 
            g_c (torch.Tensor): generated contours
            mask (torch.Tensor): mask for each note of the sample

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: generator loss, pitch loss,  loudness loss
        """

        e_wav = c2wav(e_c, self.inv_transform, self.ddsp)
        g_wav = c2wav(g_c, self.inv_transform, self.ddsp)

        disc_e = self.disc(e_wav)
        disc_g = self.disc(g_wav)

        G_loss = 0
        for scale_e, scale_g in zip(disc_e, disc_g):
            G_loss += self.criteron.g_loss(scale_e[-1], scale_g[-1])

        if self.reg:
            u_f0, u_lo = u_c.split(1, 1)
            g_f0, g_lo = g_c.split(1, 1)

            # apply inverse transform to compare pitches (midi range) and loudness (loudness range)
            inv_u_f0, inv_u_lo = self.train_set.inverse_transform(u_f0, u_lo)
            inv_g_f0, inv_g_lo = self.train_set.inverse_transform(g_f0, g_lo)

            # add pitch loss
            f0_loss, lo_loss = self.midi_loss(inv_g_f0, inv_u_f0, inv_g_lo,
                                              inv_u_lo, mask)

        else:
            f0_loss = lo_loss = 0

        return G_loss, f0_loss, lo_loss

    def disc_step(self, u_c: torch.Tensor, e_c: torch.Tensor,
                  g_c: torch.Tensor) -> torch.Tensor:
        """Compute discriminator step with expressive contours as real data and generated
        contours as fake ones.

        Args:
            u_c (torch.Tensor): unexpressive contours
            e_c (torch.Tensor): expressive contours
            g_c (torch.Tensor): generated contours

        Returns:
            torch.Tensor: dicriminator loss according to criteron
        """

        e_wav = c2wav(e_c, self.inv_transform, self.ddsp)
        g_wav = c2wav(g_c, self.inv_transform, self.ddsp)

        # discriminate

        disc_e = self.disc(e_wav)
        disc_g = self.disc(g_wav.detach())

        D_loss = 0

        for scale_e, scale_g in zip(disc_e, disc_g):
            D_loss += self.criteron.d_loss(scale_e[-1], scale_g[-1])

        return D_loss

    def training_step(self, batch: List[torch.Tensor],
                      batch_idx: int) -> OrderedDict:
        """Compute a training step for generator or discriminator 
        (according to optimizer index)

        Args:
            batch (List[torch.Tensor]): batch composed of (u_c, e_c, onsets, offsets)
            batch_idx (int): batch index

        Returns:
            OrderedDict: dict {loss, progress_bar, log}
        """

        g_opt, d_opt = self.optimizers()

        u_f0, u_lo, e_f0, e_lo, onsets, offsets, mask = batch

        u_c = torch.cat([u_f0, u_lo], -2)
        e_c = torch.cat([e_f0, e_lo], -2)

        # generate new contours
        g_c = self(u_c)

        # train discriminator
        D_loss = self.disc_step(u_c, e_c, g_c)

        d_opt.zero_grad()
        self.manual_backward(D_loss)
        d_opt.step()

        # train generator

        G_loss, f0_loss, lo_loss = self.gen_step(u_c, e_c, g_c, mask)

        # we train the generator alternatively on the adversarial objective and
        # the accuracy of the generated notes
        g_opt.zero_grad()
        self.manual_backward(G_loss)  # + f0_loss + lo_loss)
        g_opt.step()

        # build contours and losses dicts
        c_dict = {"u": u_c, "e": e_c, "g": g_c}
        loss_dict = {"G": G_loss, "D": D_loss, "f0": f0_loss, "lo": lo_loss}

        logging(self.log, self.logger, "train", self.train_idx, self.reg,
                c_dict, loss_dict, self.inv_transform, self.ddsp)

        self.train_idx += 1

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Compute validation step (do some logging)

        Args:
            batch (torch.Tensor): batch composed of (u_c, e_c, onsets, offsets)
            batch_idx (int): batch index
        """

        u_f0, u_lo, e_f0, e_lo, onsets, offsets, mask = batch

        u_c = torch.cat([u_f0, u_lo], -2)
        e_c = torch.cat([e_f0, e_lo], -2)

        # generate new contours
        g_c = self(u_c)
        D_loss = self.disc_step(u_c, e_c, g_c)
        G_loss, f0_loss, lo_loss = self.gen_step(u_c, e_c, g_c, mask)

        c_dict = {"u": u_c, "e": e_c, "g": g_c}
        loss_dict = {"G": G_loss, "D": D_loss, "f0": f0_loss, "lo": lo_loss}

        logging(self.log, self.logger, "val", self.train_idx, self.reg, c_dict,
                loss_dict, self.inv_transform, self.ddsp)

        self.val_idx += 1

    def configure_optimizers(self) -> Tuple:
        """Configure both generator and discriminator optimizers

        Returns:
            Tuple(list): (list of optimizers, empty list) 
        """

        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.gen.parameters(),
                                 lr=lr)  #, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.disc.parameters(),
                                 lr=lr)  #, betas=(b1, b2))
        #opt_d = torch.optim.SGD(self.disc.parameters(), lr=lr)

        return opt_g, opt_d

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set,
                          batch_size=8,
                          shuffle=True,
                          num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=8,
                          shuffle=False,
                          num_workers=8)


if __name__ == "__main__":
    # get dataset
    list_transforms = [(PitchTransform, {
        "feature_range": (-1, 1)
    }), (LoudnessTransform, {
        "feature_range": (-1, 1)
    })]
    n_sample = 1024
    lr = 1e-3
    criteron = Hinge_loss()
    # init model
    model = PerfGAN(g_down_channels=[2, 16, 64, 128, 512],
                    g_up_channels=[1024, 512, 128, 64, 16, 2],
                    g_down_dilations=[1, 1, 1, 1, 1],
                    g_up_dilations=[1, 1, 1, 1, 1, 1],
                    d_conv_channels=[2, 64, 128, 256, 512, 1024],
                    d_dilations=[1, 1, 1, 1, 1],
                    d_h_dims=[n_sample, 512, 128, 64, 1],
                    criteron=criteron,
                    regularization=True,
                    lr=lr,
                    b1=0.5,
                    b2=0.999)

    model.ddsp = torch.jit.load("ddsp_violin.ts").eval()

    tb_logger = pl_loggers.TensorBoardLogger('runs/')
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=10000,
        logger=tb_logger,
    )
    #log_every_n_steps=10)

    trainer.fit(model)
