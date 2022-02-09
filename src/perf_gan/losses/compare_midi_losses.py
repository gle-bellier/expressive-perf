import pytest
import torch
from torch.utils.data import DataLoader
from perf_gan.losses.midi_loss import MidiLoss
from perf_gan.data.synth_dataset import GANDataset
import matplotlib.pyplot as plt

dataset = GANDataset("data/dataset.pickle", n_sample=1024)
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

losses_mean = []
losses_prop = []
losses_both = []

N = 10

shifts = torch.arange(0, 3, 0.1)
for shift in shifts:

    m_losses_mean = 0
    m_losses_prop = 0
    m_losses_both = 0
    for i in range(N):
        sample = next(iter(dataloader))

        u_contours, e_contours, onsets, offsets = sample

        # we suppose we generate same contours as MIDI + shift
        u_f0, u_lo = u_contours.split(1, 1)
        e_f0, e_lo = e_contours.split(1, 1)

        gen_f0 = e_f0 + shift
        gen_lo = u_lo

        midi_loss = MidiLoss()
        m_losses_mean += midi_loss(gen_f0,
                                   u_f0,
                                   gen_lo,
                                   u_lo,
                                   onsets,
                                   offsets,
                                   types=["mean", "mean"])[0] / N

        m_losses_prop += midi_loss(gen_f0,
                                   u_f0,
                                   gen_lo,
                                   u_lo,
                                   onsets,
                                   offsets,
                                   types=["prop", "prop"])[0] / N
        m_losses_both += midi_loss(gen_f0,
                                   u_f0,
                                   gen_lo,
                                   u_lo,
                                   onsets,
                                   offsets,
                                   types=["both", "both"])[0] / N
    losses_mean += [m_losses_mean]
    losses_prop += [m_losses_prop]
    losses_both += [m_losses_both]

plt.plot(shifts, losses_mean, label="mean")
plt.plot(shifts, losses_prop, label="prop")
plt.plot(shifts, losses_both, label="both")

plt.legend()
plt.show()
