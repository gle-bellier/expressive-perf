import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import List, Tuple

from perf_gan.models.generator import Generator
from perf_gan.models.discriminator import Discriminator

from perf_gan.data.dataset import GANDataset
from perf_gan.losses.lsgan_loss import LSGAN_loss
from perf_gan.losses.hinge_loss import Hinge_loss
from perf_gan.losses.pitch_loss import PitchLoss


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
        self.val_idx = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute generator pass forward with unexpressive contours
        (MIDI)

        Args:
            x (torch.Tensor): unexpressive contours (B, C, L)

        Returns:
            torch.Tensor: generated expressive contours (B, C, L)
        """
        return self.gen(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int,
                      optimizer_idx: int) -> OrderedDict:
        """Compute a training step for generator or discriminator 
        (according to optimizer index)

        Args:
            batch (torch.Tensor): batch composed of (u_contours, e_contours, onsets, offsets)
            batch_idx (int): batch index
            optimizer_idx (int): optimizer index (0 for generator, 1 for discriminator)

        Returns:
            OrderedDict: dict {loss, progress_bar, log}
        """
        u_contours, e_contours, _, _ = batch

        disc_e = self.disc(e_contours).view(-1)
        fake_contours = self.gen(u_contours)

        if optimizer_idx == 1:
            # train discriminator
            disc_u = self.disc(fake_contours.detach()).view(-1)
            disc_loss = self.criteron.disc_loss(disc_e, disc_u)

            self.log("disc_loss", disc_loss)

            tqdm_dict = {'d_loss': disc_loss}
            output = OrderedDict({
                'loss': disc_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })

            return output

        if optimizer_idx == 0:
            # train generator
            disc_gu = self.disc(fake_contours).view(-1)
            gen_loss = self.criteron.gen_loss(disc_e, disc_gu)

            # add pitch loss
            pitch_loss = PitchLoss

            self.log("gen_loss", gen_loss)
            tqdm_dict = {'d_loss': gen_loss}
            output = OrderedDict({
                'loss': gen_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

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

    def configure_optimizers(self) -> Tuple[list, list]:
        """Configure both generator and discriminator optimizers

        Returns:
            Tuple(list): (list of optimizers, empty list) 
        """
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []


if __name__ == "__main__":
    # get dataset
    list_transforms = [(MinMaxScaler, {
        "feature_range": (-1, 1)
    }), (MinMaxScaler, {
        "feature_range": (-1, 1)
    })]
    dataset = GANDataset(path="data/dataset.pickle",
                         n_sample=1024,
                         list_transforms=list_transforms)
    dataset.transform()
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    criteron = Hinge_loss()

    # init model
    model = PerfGAN(g_down_channels=[2, 4, 8, 16],
                    g_up_channels=[32, 16, 8, 4, 2],
                    g_down_dilations=[1, 1, 3, 3],
                    g_up_dilations=[3, 3, 1, 1, 1],
                    d_conv_channels=[2, 32, 16, 1],
                    d_dilations=[1, 1, 3],
                    d_h_dims=[1024, 32, 1],
                    criteron=criteron,
                    lr=1e-3,
                    b1=0.999,
                    b2=0.999)

    tb_logger = pl_loggers.TensorBoardLogger('runs/')
    trainer = pl.Trainer(gpus=1, max_epochs=1000, logger=tb_logger)

    trainer.fit(model, dataloader, dataloader)
