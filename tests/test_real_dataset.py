import pytest
import torch
import numpy as np
import pickle

from perf_gan.data.preprocess import Identity, PitchTransform, LoudnessTransform
from perf_gan.data.contours_dataset import ContoursDataset


def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass


def test_dataset_items_shape():
    path = "data/dataset.pickle"

    for c in read_from_pickle(path):
        # we want to check that dimensions are equal

        assert c["u_f0"].shape == c["u_lo"].shape == c["e_f0"].shape == c[
            "e_lo"].shape == c["onsets"].shape == c["offsets"].shape == c[
                "mask"][0].shape


def test_dataset_items():
    """Test size of dataset items
    """
    l = [(PitchTransform, {
        "feature_range": (-1, 1)
    }), (LoudnessTransform, {
        "feature_range": (-1, 1)
    })]

    d = ContoursDataset(path="data/dataset.pickle", list_transforms=l)
    # loop over the 4 components (u contours, e contours, onsets, offsets)
    for idx in range(len(d)):
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
    d = ContoursDataset(path="data/dataset.pickle", list_transforms=l)

    for i in range(len(d)):
        u_f0, u_lo, e_f0, e_lo, _, _, _ = d[i]

        assert torch.min(u_f0) >= -1 and torch.max(u_f0) <= 1
        assert torch.min(u_lo) >= -1 and torch.max(u_lo) <= 1
        assert torch.min(e_f0) >= -1 and torch.max(e_f0) <= 1
        assert torch.min(e_lo) >= -1 and torch.max(e_lo) <= 1
