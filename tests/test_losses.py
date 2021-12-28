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
        (1, 0.1),
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

    u_contours, e_contours, onsets, offsets = sample

    # we suppose we generate same contours as MIDI + shift
    u_f0, _ = u_contours.split(1, 1)
    gen_f0 = u_f0 + shift

    pitch_loss = PitchLoss()
    assert pitch_loss(gen_f0, u_f0, onsets, offsets) >= expected
