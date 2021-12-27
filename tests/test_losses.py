import pytest
import torch
from torch.utils.data import DataLoader
from perf_gan.losses.pitch_loss import PitchLoss
from perf_gan.data.dataset import GANDataset
import matplotlib.pyplot as plt


@pytest.mark.parametrize(
    "shift, expected",
    [
        (0, 0),
        (.8, .8),
    ],
)
def test_pitch_loss(shift: float, expected: float) -> None:
    """Test pitch loss

    Args:
        shift (float): shift (in tone)
        expected (float): expected loss lower bound
    """

    dataset = GANDataset("data/dataset.pickle")
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    sample = next(iter(dataloader))

    u_contours, _, _, _ = sample

    # we suppose we generate same contours as MIDI
    gen_f0, _ = u_contours.split(1, 1)

    pitch_loss = PitchLoss()
    assert pitch_loss(gen_f0 + shift, sample) >= expected
