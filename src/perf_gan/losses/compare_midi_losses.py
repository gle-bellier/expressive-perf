import pytest
import torch
from torch.utils.data import DataLoader
from perf_gan.losses.midi_loss import Midi_loss

from perf_gan.data.preprocess import Identity

from perf_gan.data.contours_dataset import ContoursDataset
import matplotlib.pyplot as plt

list_transforms = [(Identity, {}), (Identity, {})]
n_sample = 1024
dataset = ContoursDataset(path="data/dataset.pickle",
                          list_transforms=list_transforms)
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

losses = []
N = 10

shifts = torch.arange(0, 3, 0.1)
for shift in shifts:

    loss = 0
    for i in range(N):
        sample = next(iter(dataloader))

        u_f0, u_lo, e_f0, e_lo, onsets, offsets, mask = sample

        gen_f0 = u_f0 + shift
        gen_lo = u_lo

        midi_loss = Midi_loss(f0_threshold=0.2, lo_threshold=3)
        loss += midi_loss(gen_f0, u_f0, gen_lo, u_lo, mask)[0] / N

    losses += [loss]

plt.plot(shifts, losses, label="mean")

plt.legend()
plt.show()
