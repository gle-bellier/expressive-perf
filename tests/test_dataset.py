import pytest
import numpy as np

from perf_gan.data.dataset_synth import Dataset


@pytest.mark.parametrize(
    "sr, n, duration",
    [
        (2000, 2, 10),
        (1600, 2, 5),
    ],
)
def test_model(sr, n, duration):
    data = Dataset(sr=sr)
    data.build(n, duration)
    print(data.e_f0)
    assert len(data.e_f0) == n * duration * sr
