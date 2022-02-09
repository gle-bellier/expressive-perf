import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from perf_gan.data.preprocess import PitchTransform, LoudnessTransform

from perf_gan.models.generator import Generator
from perf_gan.models.discriminator import Discriminator

from perf_gan.data.synth_dataset import GANDataset
from perf_gan.losses.lsgan_loss import LSGAN_loss
from perf_gan.losses.hinge_loss import Hinge_loss

writer = SummaryWriter()
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print("Training of ", device)

# define batchsize
batch_size = 32

# get dataset
list_transforms = [(PitchTransform, {
    "feature_range": (-1, 1)
}), (LoudnessTransform, {
    "feature_range": (-1, 1)
})]
dataset = GANDataset(path="data/dataset.pickle",
                     n_sample=1024,
                     list_transforms=list_transforms)

data = DataLoader(dataset=dataset, batch_size=32, shuffle=True)


def training(num_epochs: int, lr: float, beta1: float) -> None:

    criteron = LSGAN_loss()
    # define both generator and discriminator
    gen = Generator(down_channels=[2, 4, 8],
                    up_channels=[16, 8, 4, 2],
                    down_dilations=[3, 5, 5],
                    up_dilations=[5, 5, 3, 3]).to(device)

    disc = Discriminator(conv_channels=[2, 4, 1],
                         dilations=[1, 3, 3],
                         h_dims=[1024, 32, 8, 1]).to(device)

    # define both optimizers
    optim_gen = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))
    optim_disc = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, 0.999))

    # initialize each block respective losses list
    gen_losses, disc_losses = [], []

    # training loop:

    for epoch in range(num_epochs):
        for i, contours in enumerate(data):
            u_contours = contours[0].to(device)
            e_contours = contours[1].to(device)

            # Training discriminator
            disc.zero_grad()

            disc_e = disc(e_contours).view(-1)

            fake_contours = gen(u_contours)
            disc_u = disc(fake_contours.detach()).view(-1)

            disc_loss = criteron.disc_loss(disc_e, disc_u)

            # backprop
            disc_loss.backward()
            optim_disc.step()

            # Training generator
            gen.zero_grad()

            disc_gu = disc(fake_contours).view(-1)
            gen_loss = criteron.gen_loss(disc_e, disc_gu)

            gen_loss.backward()
            optim_gen.step()

            # Add to loss lists and print losses
            gen_losses += [gen_loss]
            disc_losses += [disc_loss]

            n_mean = 100
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' %
                      (epoch, num_epochs, i, len(dataset),
                       torch.mean(torch.Tensor(disc_losses[-n_mean:])),
                       torch.mean(torch.Tensor(gen_losses[-n_mean:]))))

                writer.add_scalar("Loss/gen", gen_loss, epoch)
                writer.add_scalar("Loss/disc", disc_loss, epoch)

            if i == 0 and epoch % 20 == 0:

                u_f0, u_lo = u_contours[0].split(1, 0)
                g_f0, g_lo = fake_contours[0].split(1, 0)

                plt.plot(u_f0.squeeze().cpu().detach(), label="u_f0")
                plt.plot(g_f0.squeeze().cpu().detach(), label="g_f0")
                plt.legend()
                writer.add_figure("pitch", plt.gcf(), epoch)

                plt.plot(u_lo.squeeze().cpu().detach(), label="u_lo")
                plt.plot(g_lo.squeeze().cpu().detach(), label="g_lo")
                plt.legend()
                writer.add_figure("lo", plt.gcf(), epoch)


if __name__ == "__main__":
    training(num_epochs=10000, lr=1e-4, beta1=0.5)
