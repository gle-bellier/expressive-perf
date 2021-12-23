import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from collections import OrderedDict

from perf_gan.models.generator import Generator
from perf_gan.models.discriminator import Discriminator

from perf_gan.data.dataset import GANDataset
from perf_gan.losses.lsgan_loss import LSGAN_loss


class PerfGAN(pl.LightningModule):
    def __init__(self, g_down_channels, g_up_channels, g_down_dilations,
                 g_up_dilations, d_conv_channels, d_dilations, d_h_dims,
                 criteron, b1: 0.999, b2: 0.999):
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

    def forward(self, x):
        return self.gen(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        u_contours, e_contours, _, _ = batch

        disc_e = self.disc(e_contours).view(-1)
        fake_contours = self.gen(u_contours)

        if optimizer_idx == 1:
            # train discriminator
            disc_u = self.disc(fake_contours.detach()).view(-1)
            disc_loss = self.criteron.disc_loss(disc_e, disc_u)
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
            tqdm_dict = {'d_loss': gen_loss}
            output = OrderedDict({
                'loss': gen_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(),
                                 lr=lr,
                                 betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(),
                                 lr=lr,
                                 betas=(b1, b2))
        return [opt_g, opt_d], []


if __name__ == "__main__":
    # get dataset
    list_transforms = [(MinMaxScaler, {}), (MinMaxScaler, {})]
    dataset = GANDataset(path="data/dataset.pickle",
                         n_sample=1024,
                         list_transforms=list_transforms)
    dataset.transform()
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    criteron = LSGAN_loss()

    # init model
    model = PerfGAN(g_down_channels=[2, 4, 8],
                    g_up_channels=[16, 8, 4, 2],
                    g_down_dilation=[3, 5, 5],
                    g_up_dilations=[5, 5, 3, 3],
                    d_conv_channels=[2, 4, 1],
                    d_dilations=[1, 3, 3],
                    d_h_dims=[1024, 32, 8, 1],
                    criteron=criteron)

    tb_logger = pl_loggers.TensorBoardLogger('runs/')
    trainer = pl.Trainer(gpus=1, max_epochs=1000, logger=tb_logger)

    trainer.fit(model, dataloader)
