import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle
import numpy as np
from typing import List, Union
from random import randint
import re

from perf_gan.data.preprocess import Identity, PitchTransform, LoudnessTransform


class SynthDataset(Dataset):
    def __init__(self, path: str, list_transforms=None, eval=False):
        """Create the Dataset object relative to the data file (given with path)

        Args:
            path (str): path to the data file
            list_transforms (list, optional): list of the transforms to be applied to the dataset.
            Defaults to None.
        """

        self.path = path
        self.dataset = open(path, "rb")

        # add transformations applied to data
        if list_transforms is None:
            self.list_transforms = [(Identity, {}), (Identity, {})]
        else:
            self.list_transforms = list_transforms

        self.eval = eval

        print("Dataset loaded.")

    def __fit_transforms(self, u_f0, u_lo, e_f0, e_lo) -> List[object]:
        """Fit the two transforms to the contours

        Returns:
            list[object]: fitted scalers
        """
        scalers = []

        # pitch
        contour = np.concatenate((u_f0, e_f0))
        transform = self.list_transforms[0]
        sc = transform[0](**transform[1]).fit(contour.reshape(-1, 1))
        scalers.append(sc)

        # loudness
        contour = np.concatenate((u_lo, e_lo))
        transform = self.list_transforms[1]
        sc = transform[0](**transform[1]).fit(contour.reshape(-1, 1))
        scalers.append(sc)

        return scalers

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformations to the contour x (pitch and loudness)

        Args:
            x (torch.Tensor): input contour

        Returns:
            torch.Tensor: transformed contour
        """

        f0, lo = torch.split(x, 1, -2)
        # transforms
        f0 = self.scalers[0].transform(f0)
        lo = self.scalers[1].transform(lo)

        return torch.cat([f0, lo], -2)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inverse transformations to the contour x (pitch and loudness)

        Args:
            x (torch.Tensor): input contour

        Returns:
            torch.Tensor: inverse transformed contour
        """

        f0, lo = torch.split(x, 1, -2)
        # Inverse transforms
        f0 = self.scalers[0].inverse_transform(f0)
        lo = self.scalers[1].inverse_transform(lo)

        return torch.cat([f0, lo], -2)

    def __len__(self) -> int:
        """Compute the number of samples in the dataset

        Returns:
            [int]: number of samples in the dataset
        """
        return int(re.split("/|_|\.", self.path)[-2])

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        """Select the ith sample from the dataset

        Args:
            idx (int): index of the sample

        Returns:
            list[torch.Tensor]: list of contours (pitch, loudness, onsets, offsets)
        """

        sample = pickle.load(self.dataset)

        u_f0 = torch.Tensor(sample["u_f0"])
        u_lo = torch.Tensor(sample["u_lo"])

        e_f0 = torch.Tensor(sample["e_f0"])
        e_lo = torch.Tensor(sample["e_lo"])

        self.scalers = self.__fit_transforms(u_f0, u_lo, e_f0, e_lo)

        s_onsets = torch.Tensor(sample["onsets"]).unsqueeze(0)
        s_offsets = torch.Tensor(sample["offsets"]).unsqueeze(0)

        mask = torch.Tensor(sample["mask"])

        # concatenate the contours into unexpressive/expressive tensors
        u_contours = torch.cat([
            u_f0.unsqueeze(0),
            u_lo.unsqueeze(0),
        ], 0)

        e_contours = torch.cat([
            e_f0.unsqueeze(0),
            e_lo.unsqueeze(0),
        ], 0)

        return [
            self.transform(u_contours),
            self.transform(e_contours), s_onsets, s_offsets, mask
        ]


if __name__ == '__main__':
    l = [(PitchTransform, {
        "feature_range": (-1, 1)
    }), (LoudnessTransform, {
        "feature_range": (-1, 1)
    })]

    d = SynthDataset(path="data/dataset_train_1000.pickle", list_transforms=l)
    # loop over the 4 components (u contours, e contours, onsets, offsets)
    for i in range(10):
        sample = d[0]
        print("New sample")
        for elt in sample:
            print(elt.shape)
