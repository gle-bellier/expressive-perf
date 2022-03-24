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
from perf_gan.models.melgan_disc import WavDiscriminator
from perf_gan.models.discriminator import Discriminator

from perf_gan.data.contours_dataset import ContoursDataset
from perf_gan.data.preprocess import PitchTransform, LoudnessTransform
from perf_gan.utils.post_processing import logging, c2wav

from perf_gan.losses.lsgan_loss import LSGAN_loss
from perf_gan.losses.hinge_loss import Hinge_loss
from perf_gan.losses.midi_loss import Midi_loss

import warnings

warnings.filterwarnings('ignore')


class PerfGAN(pl.LightningModule):

    def __init__(self,
                 g_params,
                 d_params,
                 d_wav_params,
                 criteron: float,
                 regularization: bool,
                 lr: float,
                 b1: int,
                 b2: int,
                 n_step_warmup=0):
        """[summary]

        Args:
            mode (str): equals to "dev" for the model generating pitch and loudness deviation or "contours" for full contours generation
            g_params
            d_params
            criteron (float): criteron for both generator and discriminator
            lr (float): learning rate
            b1 (int): b1 factor 
            b2 (int): b2 factor

        """
        super(PerfGAN, self).__init__()

        self.save_hyperparameters()

        self.gen = Generator(channels=g_params["channels"], dropout=0.)

        self.disc = Discriminator(channels=d_params["channels"],
                                  n_layers=d_params["n_layers"],
                                  n_sample=512,
                                  dropout=0.5)

        self.wav_disc = WavDiscriminator(
            num_D=d_wav_params["num_D"],
            ndf=d_wav_params["ndf"],
            n_layers=d_wav_params["n_layers"],
            downsampling_factor=d_wav_params["down_factor"])

        self.criteron = criteron
        self.reg = regularization

        self.n_step_warmup = n_step_warmup
        self.warmup = False

        self.midi_loss = Midi_loss(f0_threshold=0.3, lo_threshold=2).cuda()

        self.train_set = ContoursDataset(path="data/train_c.pickle",
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

        # generate contours deviation

        noise = torch.randn_like(x)
        noisy = torch.cat([x, noise], dim=1)

        return self.gen(noisy) + x

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
        if self.warmup:
            disc_e = self.disc(e_c)
            disc_g = self.disc(g_c)
            G_loss = self.criteron.g_loss(disc_e, disc_g)
        else:
            e_wav = c2wav(self, e_c)
            g_wav = c2wav(self, g_c)

            disc_e = self.wav_disc(e_wav)
            disc_g = self.wav_disc(g_wav)

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

        if self.warmup:
            disc_e = self.disc(e_c)
            disc_g = self.disc(g_c.detach())

            D_loss = self.criteron.d_loss(disc_e, disc_g)

        else:
            e_wav = c2wav(self, e_c)
            g_wav = c2wav(self, g_c)

            # discriminate

            disc_e = self.wav_disc(e_wav)
            disc_g = self.wav_disc(g_wav.detach())

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

        self.warmup = self.train_idx < self.n_step_warmup
        g_opt, d_opt, d_wav_opt = self.optimizers()

        u_f0, u_lo, e_f0, e_lo, _, _, mask = batch

        u_c = torch.cat([u_f0, u_lo], -2)
        e_c = torch.cat([e_f0, e_lo], -2)

        # generate new contours
        g_c = self(u_c)

        # train discriminator
        D_loss = self.disc_step(u_c, e_c, g_c)

        if self.warmup:
            d_opt.zero_grad()
            self.manual_backward(D_loss)
            d_opt.step()

        else:
            d_wav_opt.zero_grad()
            self.manual_backward(D_loss)
            d_wav_opt.step()

        # train generator

        G_loss, f0_loss, lo_loss = self.gen_step(u_c, e_c, g_c, mask)

        # we train the generator alternatively on the adversarial objective and
        # the accuracy of the generated notes

        g_opt.zero_grad()

        if self.train_idx % 2:
            self.manual_backward(G_loss)
        else:
            self.manual_backward(f0_loss + lo_loss)
        g_opt.step()

        # build contours and losses dicts
        c_dict = {"u": u_c, "e": e_c, "g": g_c}
        loss_dict = {"G": G_loss, "D": D_loss, "f0": f0_loss, "lo": lo_loss}

        logging(self, "train", c_dict, loss_dict)

        self.train_idx += 1

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Compute validation step (do some logging)

        Args:
            batch (torch.Tensor): batch composed of (u_c, e_c, onsets, offsets)
            batch_idx (int): batch index
        """

        u_f0, u_lo, e_f0, e_lo, _, _, mask = batch

        u_c = torch.cat([u_f0, u_lo], -2)
        e_c = torch.cat([e_f0, e_lo], -2)

        # generate new contours
        g_c = self(u_c)
        D_loss = self.disc_step(u_c, e_c, g_c)
        G_loss, f0_loss, lo_loss = self.gen_step(u_c, e_c, g_c, mask)

        c_dict = {"u": u_c, "e": e_c, "g": g_c}
        loss_dict = {"G": G_loss, "D": D_loss, "f0": f0_loss, "lo": lo_loss}

        logging(self, "val", c_dict, loss_dict)

        self.val_idx += 1

    def set_ddsp(self, ddsp):
        self.ddsp = ddsp
        # freeze ddsp:
        for p in self.ddsp.parameters():
            p.requires_grad = False

    def configure_optimizers(self) -> Tuple:
        """Configure both generator and discriminator optimizers

        Returns:
            Tuple(list): (list of optimizers, empty list) 
        """

        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(b1, b2))
        opt_d_wav = torch.optim.Adam(self.wav_disc.parameters(),
                                     lr=lr,
                                     betas=(b1, b2))

        opt_d = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(b1, b2))
        # opt_d = torch.optim.SGD(self.disc.parameters(), lr=lr)

        return opt_g, opt_d, opt_d_wav

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set,
                          batch_size=16,
                          shuffle=True,
                          num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=16,
                          shuffle=False,
                          num_workers=8)


if __name__ == "__main__":
    # get dataset
    list_transforms = [(PitchTransform, {
        "feature_range": (-1, 1)
    }), (LoudnessTransform, {
        "feature_range": (-1, 1)
    })]
    n_sample = 512
    lr = 1e-3
    criteron = LSGAN_loss()  #Hinge_loss()

    g_params = {
        "channels": [4, 16, 64, 128, 512, 1024],
    }

    d_params = {"channels": [2, 32, 64, 512, 1024], "n_layers": 5}
    d_wav_params = {"num_D": 4, "ndf": 16, "n_layers": 4, "down_factor": 4}

    # init model
    model = PerfGAN(g_params,
                    d_params,
                    d_wav_params,
                    criteron=criteron,
                    regularization=True,
                    lr=lr,
                    b1=0.5,
                    b2=0.999,
                    n_step_warmup=-1)

    model.set_ddsp(torch.jit.load("ddsp_violin.ts"))

    tb_logger = pl_loggers.TensorBoardLogger('runs/',
                                             name="perf-gan",
                                             default_hp_metric=False)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=10000,
        logger=tb_logger,
    )
    #log_every_n_steps=10)

    trainer.fit(model)
