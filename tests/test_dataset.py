import pytest
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from perf_gan.data.make_dataset import DatasetCreator
from perf_gan.data.dataset import GANDataset


@pytest.mark.parametrize(
    "sr, n, duration",
    [
        (2048, 2, 10),
        (1024, 2, 5),
    ],
)
def test_dataset_creation(sr: int, n: int, duration: int):
    """Check dataset creation

    Args:
        sr (int): sampling rate
        n (int): number of samples in the dataset
        duration (int): length of each sample
    """
    data = DatasetCreator(sr=sr)
    data.build(n, duration)
    assert len(data.e_f0) == n * duration * sr


def test_dataset_items():
    """Test size of dataset items
    """
    l = [(StandardScaler, {}), (MinMaxScaler, {})]
    for size in [2048, 1024, 512]:
        d = GANDataset(path="data/dataset.pickle",
                       n_sample=size,
                       list_transforms=l)
        d.transform()
        # loop over the 4 components (u contours, e contours, onsets, offsets)
        item = d[0]
        for idx in range(4):
            assert len(item[idx]) == size


def test_dataset_items_range():
    """Test size of dataset items ranges
    """
    l = [(StandardScaler, {}), (MinMaxScaler, {})]
    d = GANDataset(path="data/dataset.pickle",
                   n_sample=1024,
                   list_transforms=l)
    d.transform()
    # loop over the 4 components (u contours, e contours, onsets, offsets)
    assert torch.min(d.e_lo) == 0 and torch.max(d.e_lo) == 1
