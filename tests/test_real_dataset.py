import pytest
import torch
import numpy as np
import pickle


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
