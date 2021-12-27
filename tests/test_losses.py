import pytest
import torch
from torch.utils.data import DataLoader
from perf_gan.losses.pitch_loss import PitchLoss
from perf_gan.data.dataset import GANDataset


def test_pitch_loss():
    """Test pitch loss
    """

    dataset = GANDataset("data/dataset.pickle")
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    sample = next(iter(dataloader))

    u_contours, e_contours, _, _ = sample

    print(u_contours.shape)
    gen_f0, _ = u_contours.split(1, 1)

    print(gen_f0.shape)

    pitch_loss = PitchLoss()
    print(pitch_loss(gen_f0, sample))


test_pitch_loss()