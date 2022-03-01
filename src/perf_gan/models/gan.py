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

from perf_gan.data.contours_dataset import ContoursDataset

from perf_gan.data.preprocess import PitchTransform, LoudnessTransform

from perf_gan.losses.lsgan_loss import LSGAN_loss
from perf_gan.losses.hinge_loss import Hinge_loss
from perf_gan.losses.midi_loss import Midi_loss

import warnings

warnings.filterwarnings('ignore')


class PerfGAN(pl.LightningModule):

    def __init__(self, mode: str, g_down_channels: List[int],
                 g_up_channels: List[int], g_down_dilations: List[int],
                 g_up_dilations: List[int], d_conv_channels: List[int],
                 d_dilations: List[int], d_h_dims: List[int], criteron: float,
                 regularization: bool, lr: float, b1: int, b2: int):
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

        assert mode in ["dev", "contours"
                        ], f"Mode should be in [dev, contours], got {mode}"
        self.mode = mode

        self.gen = Generator(down_channels=g_down_channels,
                             up_channels=g_up_channels,
                             down_dilations=g_down_dilations,
                             up_dilations=g_up_dilations)

        self.disc = Discriminator(conv_channels=d_conv_channels,
                                  dilations=d_dilations,
                                  h_dims=d_h_dims)

        self.criteron = criteron
        self.reg = regularization
        self.dataset = None
        self.midi_loss = Midi_loss(f0_threshold=0.3, lo_threshold=2).cuda()

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

        # generate deviations or whole contours

        if self.mode == "dev":
            return self.gen(x) + x
        else:
            return self.gen(x)

    def gen_step(
            self, u_contours: torch.Tensor, e_contours: torch.Tensor,
            gen_contours: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute generator loss according to criteron.
        Expressive contours are considered the real data. In case of pitch and loudness 
        regularization this function outputs non null pitch and loudness proper losses

        Args:
            u_contours (torch.Tensor): unexpressive contours
            e_contours (torch.Tensor): expressive contours 
            gen_contours (torch.Tensor): generated contours
            mask (torch.Tensor): mask for each note of the sample

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: generator loss, pitch loss,  loudness loss
        """

        disc_e = self.disc(e_contours).view(-1)
        disc_gu = self.disc(gen_contours).view(-1)

        gen_loss = self.criteron.gen_loss(disc_e, disc_gu)

        if self.reg:
            u_f0, u_lo = u_contours.split(1, 1)
            gen_f0, gen_lo = gen_contours.split(1, 1)

            # apply inverse transform to compare pitches (midi range) and loudness (loudness range)
            inv_u_f0, inv_u_lo = self.dataset.inverse_transform(u_f0, u_lo)
            inv_gen_f0, inv_gen_lo = self.dataset.inverse_transform(
                gen_f0, gen_lo)

            # add pitch loss
            pitch_loss, lo_loss = self.midi_loss(inv_gen_f0, inv_u_f0,
                                                 inv_gen_lo, inv_u_lo, mask)

        else:
            pitch_loss = lo_loss = 0

        return gen_loss, pitch_loss, lo_loss

    def disc_step(self, u_contours: torch.Tensor, e_contours: torch.Tensor,
                  gen_contours: torch.Tensor) -> torch.Tensor:
        """Compute discriminator step with expressive contours as real data and generated
        contours as fake ones.

        Args:
            u_contours (torch.Tensor): unexpressive contours
            e_contours (torch.Tensor): expressive contours
            gen_contours (torch.Tensor): generated contours

        Returns:
            torch.Tensor: dicriminator loss according to criteron
        """

        # discriminate
        disc_e = self.disc(e_contours).view(-1)
        disc_u = self.disc(gen_contours.detach()).view(-1)

        disc_loss = self.criteron.disc_loss(disc_e, disc_u)

        return disc_loss

    def training_step(self, batch: List[torch.Tensor],
                      batch_idx: int) -> OrderedDict:
        """Compute a training step for generator or discriminator 
        (according to optimizer index)

        Args:
            batch (List[torch.Tensor]): batch composed of (u_contours, e_contours, onsets, offsets)
            batch_idx (int): batch index

        Returns:
            OrderedDict: dict {loss, progress_bar, log}
        """

        g_opt, d_opt = self.optimizers()

        u_f0, u_lo, e_f0, e_lo, onsets, offsets, mask = batch

        u_contours = torch.cat([u_f0, u_lo], -2)
        e_contours = torch.cat([e_f0, e_lo], -2)

        # generate new contours
        gen_contours = self(u_contours)

        # train discriminator
        disc_loss = self.disc_step(u_contours, e_contours, gen_contours)

        self.disc.zero_grad()
        self.manual_backward(disc_loss)
        d_opt.step()

        # train generator

        gen_loss, pitch_loss, lo_loss = self.gen_step(u_contours, e_contours,
                                                      gen_contours, mask)

        g_opt.zero_grad()
        self.manual_backward(gen_loss + 10 * pitch_loss + lo_loss)
        g_opt.step()

        if self.reg:
            self.log("reg/f0_loss", pitch_loss)
            self.log("reg/lo_loss", lo_loss)

        self.logger.experiment.add_scalars(
            'abversarial',
            {
                'gen': gen_loss,
                'disc': disc_loss
            },
            global_step=self.train_idx,
        )

        self.train_idx += 1

    def __midi2hz(self, x):
        return torch.pow(2, (x - 69) / 12) * 440

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Compute validation step (do some logging)

        Args:
            batch (torch.Tensor): batch composed of (u_contours, e_contours, onsets, offsets)
            batch_idx (int): batch index
        """
        self.val_idx += 1
        if self.val_idx % 10 == 0:

            u_f0, u_lo, e_f0, e_lo, onsets, offsets, mask = batch

            u_contours = torch.cat([u_f0, u_lo], -2)
            e_contours = torch.cat([e_f0, e_lo], -2)
            gen_contours = self(u_contours)

            u_f0, u_lo = u_contours[0].split(1, -2)
            e_f0, e_lo = e_contours[0].split(1, -2)
            g_f0, g_lo = gen_contours[0].split(1, -2)

            # apply inverse transform

            u_f0, u_lo = self.dataset.inverse_transform(u_f0, u_lo)
            e_f0, e_lo = self.dataset.inverse_transform(e_f0, e_lo)
            g_f0, g_lo = self.dataset.inverse_transform(g_f0, g_lo)

            # convert midi to hz / db

            u_f0 = self.__midi2hz(u_f0[0])
            e_f0 = self.__midi2hz(e_f0[0])
            g_f0 = self.__midi2hz(g_f0[0])

            if self.reg:
                plt.plot(u_f0.squeeze().cpu().detach(), label="u_f0")
            plt.plot(g_f0.squeeze().cpu().detach(), label="g_f0")
            plt.legend()
            self.logger.experiment.add_figure("contours/gen/f0", plt.gcf(),
                                              self.val_idx)

            if self.reg:
                plt.plot(e_f0.squeeze().cpu().detach(), label="e_f0")
                plt.legend()
                self.logger.experiment.add_figure("contours/sample/f0",
                                                  plt.gcf(), self.val_idx)

            if self.reg:
                plt.plot(u_lo.squeeze().cpu().detach(), label="u_lo")
            plt.plot(g_lo.squeeze().cpu().detach(), label="g_lo")
            plt.legend()
            self.logger.experiment.add_figure("contours/gen/lo", plt.gcf(),
                                              self.val_idx)

            if self.reg:
                plt.plot(e_lo.squeeze().cpu().detach(), label="e_lo")
                plt.legend()
                self.logger.experiment.add_figure("contours/sample/lo",
                                                  plt.gcf(), self.val_idx)

            if self.ddsp is not None:
                g_f0 = g_f0.float().reshape(1, -1, 1)
                # artificialy add 3db
                g_lo = g_lo.float().reshape(1, -1, 1) + 3
                signal = self.ddsp(g_f0, g_lo)
                signal = signal.reshape(-1).cpu().numpy()
                self.logger.experiment.add_audio(
                    "generation",
                    signal,
                    self.val_idx,
                    16000,
                )

    def configure_optimizers(self) -> Tuple:
        """Configure both generator and discriminator optimizers

        Returns:
            Tuple(list): (list of optimizers, empty list) 
        """

        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(b1, b2))
        #opt_d = torch.optim.SGD(self.disc.parameters(), lr=lr, momentum=.5)

        return opt_g, opt_d


if __name__ == "__main__":
    # get dataset
    list_transforms = [(PitchTransform, {
        "feature_range": (-1, 1)
    }), (LoudnessTransform, {
        "feature_range": (-1, 1)
    })]
    n_sample = 1024
    train_set = ContoursDataset(path="data/dataset_aug.pickle",
                                list_transforms=list_transforms)
    train_dataloader = DataLoader(dataset=train_set,
                                  batch_size=64,
                                  shuffle=True,
                                  num_workers=8)
    test_set = ContoursDataset(path="data/dataset_aug.pickle",
                               list_transforms=list_transforms)
    test_dataloader = DataLoader(dataset=test_set,
                                 batch_size=64,
                                 shuffle=True,
                                 num_workers=8)

    lr = 1e-3
    criteron = Hinge_loss()
    # init model
    model = PerfGAN(mode="dev",
                    g_down_channels=[2, 32, 64, 128],
                    g_up_channels=[512, 128, 64, 32, 2],
                    g_down_dilations=[3, 1, 1, 1],
                    g_up_dilations=[3, 1, 1, 1, 1],
                    d_conv_channels=[2, 64, 1024, 512, 32, 1],
                    d_dilations=[1, 1, 1, 1, 1, 1],
                    d_h_dims=[n_sample, 512, 64, 1],
                    criteron=criteron,
                    regularization=False,
                    lr=lr,
                    b1=0.5,
                    b2=0.999)

    model.dataset = train_set
    model.ddsp = torch.jit.load("ddsp_violin.ts").eval()

    tb_logger = pl_loggers.TensorBoardLogger('runs/')
    trainer = pl.Trainer(gpus=1,
                         max_epochs=10000,
                         logger=tb_logger,
                         log_every_n_steps=10)

    trainer.fit(model, train_dataloader, test_dataloader)
