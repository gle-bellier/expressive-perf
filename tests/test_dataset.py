import pytest
import torch
import numpy as np

from perf_gan.data.make_dataset import DatasetCreator

from perf_gan.data.preprocess import PitchTransform, LoudnessTransform
from perf_gan.data.dataset import GANDataset


def test_dataset_items():
    """Test size of dataset items
    """
    l = [(PitchTransform, {
        "feature_range": (-1, 1)
    }), (LoudnessTransform, {
        "feature_range": (-1, 1)
    })]

    d = GANDataset(path="data/dataset_train_1000.pickle", list_transforms=l)
    # loop over the 4 components (u contours, e contours, onsets, offsets)
    for idx in range(4):
        item = d[idx]
        assert (item[0].shape[-1]) == 1024


def test_dataset_items_range():
    """Test size of dataset items ranges
    """
    l = [(PitchTransform, {
        "feature_range": (-1, 1)
    }), (LoudnessTransform, {
        "feature_range": (-1, 1)
    })]
    d = GANDataset(path="data/dataset_train_1000.pickle", list_transforms=l)

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
    d = GANDataset(path="data/dataset_train_1000.pickle")

    n = 0
    for i in range(50):
        contour = d[i][0]
        t_contour = d.transform(contour)
        i_contour = d.inverse_transform(t_contour)
        n += torch.allclose(contour, i_contour)
    assert n == 50