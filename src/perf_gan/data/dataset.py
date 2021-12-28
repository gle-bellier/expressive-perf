import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle
import numpy as np
from typing import List, Union
from random import randint

from perf_gan.data.preprocess import Identity


class GANDataset(Dataset):
    def __init__(self,
                 path: str,
                 n_sample=2048,
                 list_transforms=None,
                 eval=False):
        """Create the Dataset object relative to the data file (given with path)

        Args:
            path (str): path to the data file
            n_sample (int, optional): length of the samples. Defaults to 2048.
            list_transforms (list, optional): list of the transforms to be applied to the dataset.
            Defaults to None.
        """

        print("Loading Dataset...")
        with open(path, "rb") as dataset:
            dataset = pickle.load(dataset)

        self.u_f0 = torch.Tensor(dataset["u_f0"])
        self.e_f0 = torch.Tensor(dataset["e_f0"])
        self.u_lo = torch.Tensor(dataset["u_lo"])
        self.e_lo = torch.Tensor(dataset["e_lo"])
        self.onsets = torch.Tensor(dataset["onsets"])
        self.offsets = torch.Tensor(dataset["offsets"])

        self.N = len(dataset["u_f0"])
        self.n_sample = n_sample

        # add transformations applied to data
        if list_transforms is None:
            self.list_transforms = [(Identity, {}), (Identity, {})]
        else:
            self.list_transforms = list_transforms

        # build scalers
        self.scalers = self.__fit_transforms()
        self.eval = eval

        print("Dataset loaded.")

    def __fit_transforms(self) -> List[object]:
        """Fit the two transforms to the contours

        Returns:
            list[object]: fitted scalers
        """
        scalers = []

        # pitch
        contour = torch.cat((self.u_f0, self.e_f0), -1)
        transform = self.list_transforms[0]
        sc = transform[0](**transform[1]).fit(contour)
        scalers.append(sc)

        # loudness
        contour = torch.cat((self.u_lo, self.e_lo), -1)
        transform = self.list_transforms[1]
        sc = transform[0](**transform[1]).fit(contour)
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
        # Inverse transforms
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
        return self.N // self.n_sample

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        """Select the ith sample from the dataset

        Args:
            idx (int): index of the sample

        Returns:
            list[torch.Tensor]: list of contours (pitch, loudness, onsets, offsets)
        """
        N = self.n_sample
        idx *= N
        # add jitter during training only
        if not self.eval:
            idx += randint(0, N // 10)

        idx = max(idx, 0)
        idx = min(idx, len(self) * self.n_sample - self.n_sample)

        # select the sample contours
        s_u_f0 = self.u_f0[idx:idx + self.n_sample]
        s_u_lo = self.u_lo[idx:idx + self.n_sample]
        s_e_f0 = self.e_f0[idx:idx + self.n_sample]
        s_e_lo = self.e_lo[idx:idx + self.n_sample]
        s_onsets = self.onsets[idx:idx + self.n_sample]
        s_offsets = self.offsets[idx:idx + self.n_sample]

        # concatenate the contours into unexpressive/expressive tensors
        u_contours = torch.cat([
            s_u_f0.unsqueeze(0),
            s_u_lo.unsqueeze(0),
        ], 0)

        e_contours = torch.cat([
            s_e_f0.unsqueeze(0),
            s_e_lo.unsqueeze(0),
        ], 0)

        return [
            self.transform(u_contours),
            self.transform(e_contours), s_onsets, s_offsets
        ]
