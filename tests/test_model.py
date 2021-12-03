import pytest
import numpy as np

from perf_gan.models.model import Model


@pytest.mark.parametrize(
    "point, expected",
    [
        (np.array([0, 0]), np.array([0, 0])),
        (np.array([2.5, 2.5]), np.array([6.25, 6.25])),
        (np.array([-1, 5]), np.array([1, 25])),
        (np.array([10, 3]), np.array([100, 9])),
    ],
)
def test_model(point, expected):
    rslt = Model().forward(point)
    assert (rslt == expected).all()
