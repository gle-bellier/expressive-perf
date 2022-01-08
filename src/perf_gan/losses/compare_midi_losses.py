import pytest
import torch
from torch.utils.data import DataLoader
from perf_gan.losses.midi_loss import MidiLoss
from perf_gan.data.dataset import GANDataset
import matplotlib.pyplot as plt

dataset = GANDataset("data/dataset.pickle")
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

sample = next(iter(dataloader))

u_contours, e_contours, onsets, offsets = sample

# we suppose we generate same contours as MIDI + shift
u_f0, u_lo = u_contours.split(1, 1)

losses_mean = []
losses_prop = []

for shift in torch.arange(0, 1, 0.1):
    gen_f0 = u_f0 + shift
    gen_lo = u_lo

    midi_loss = MidiLoss()
    losses_mean += [
        midi_loss(gen_f0,
                  u_f0,
                  gen_lo,
                  u_lo,
                  onsets,
                  offsets,
                  types=["mean", "mean"])
    ]
    losses_prop += [
        midi_loss(gen_f0,
                  u_f0,
                  gen_lo,
                  u_lo,
                  onsets,
                  offsets,
                  types=["prop", "prop"])
    ]
