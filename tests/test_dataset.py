import pytest
import numpy as np

from perf_gan.data.make_dataset import DatasetCreator
from perf_gan.data.dataset import GANDataset


@pytest.mark.parametrize(
    "sr, n, duration",
    [
        (2048, 2, 10),
        (1024, 2, 5),
    ],
)
def test_model(sr, n, duration):
    data = DatasetCreator(sr=sr)
    data.build(n, duration)
    print(data.e_f0)
    assert len(data.e_f0) == n * duration * sr
