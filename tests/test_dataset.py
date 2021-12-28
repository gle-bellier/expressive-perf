import pytest
import torch
import numpy as np

from perf_gan.data.make_dataset import DatasetCreator

from perf_gan.data.preprocess import PitchTransform, LoudnessTransform
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
    l = [(PitchTransform, {
        "feature_range": (-1, 1)
    }), (LoudnessTransform, {
        "feature_range": (-1, 1)
    })]
    for size in [2048, 1024, 512]:
        d = GANDataset(path="data/dataset.pickle",
                       n_sample=size,
                       list_transforms=l)
        # loop over the 4 components (u contours, e contours, onsets, offsets)
        item = d[0]
        for idx in range(4):
            assert (item[idx].shape[-1]) == size


def test_dataset_items_range():
    """Test size of dataset items ranges
    """
    l = [(PitchTransform, {
        "feature_range": (-1, 1)
    }), (LoudnessTransform, {
        "feature_range": (-1, 1)
    })]
    d = GANDataset(path="data/dataset.pickle",
                   n_sample=1024,
                   list_transforms=l)

    e_lo = d[0][1][1]
    assert torch.min(e_lo) >= -1 and torch.max(e_lo) <= 1


def test_dataset_inv_transform():
    """Test inverse transforms (with the 50 first samples)
    """

    l = [(PitchTransform, {
        "feature_range": (-1, 1)
    }), (LoudnessTransform, {
        "feature_range": (-1, 1)
    })]
    d = GANDataset(path="data/dataset.pickle",
                   n_sample=1024,
                   list_transforms=l)

    id = GANDataset(path="data/dataset.pickle", n_sample=1024)

    n = 0
    for i in range(50):
        contour = id[i][0]
        t_contour = d.transform(contour)
        i_contour = d.inverse_transform(t_contour)
        n += torch.allclose(contour, i_contour)
    assert n == 50