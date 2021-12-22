import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle
import numpy as np
from typing import List, Union
from random import randint


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

        self.u_f0 = dataset["u_f0"]
        self.e_f0 = dataset["e_f0"]
        self.u_lo = dataset["u_lo"]
        self.e_lo = dataset["e_lo"]
        self.onsets = dataset["onsets"]
        self.offsets = dataset["offsets"]

        self.N = len(dataset["u_f0"])
        self.n_sample = n_sample
        self.list_transforms = list_transforms
        self.scalers = None
        self.eval = eval

        print("Dataset loaded.")

    def __fit_transforms(self) -> List[object]:
        """Fit the two transforms to the contours

        Returns:
            list[object]: fitted scalers
        """
        scalers = []

        # pitch
        contour = np.concatenate((self.u_f0, self.e_f0))
        transform = self.list_transforms[0]
        sc = transform[0](**transform[1]).fit(contour.reshape(-1, 1))
        scalers.append(sc)

        # loudness
        contour = np.concatenate((self.u_lo, self.e_lo))
        transform = self.list_transforms[1]
        sc = transform[0](**transform[1]).fit(contour.reshape(-1, 1))
        scalers.append(sc)

        return scalers

    def transform(self) -> None:
        """Transform the dataset contours according to the scalers
        """

        # fit transforms to contours
        self.scalers = self.__fit_transforms()

        # convert onsets and offsets to torch arrays
        self.onsets = torch.from_numpy(self.onsets).float()
        self.offsets = torch.from_numpy(self.offsets).float()

        # apply transforms to pitch and loudness contours

        self.u_f0 = self.__apply_transform(self.u_f0, self.scalers[0])
        self.e_f0 = self.__apply_transform(self.e_f0, self.scalers[0])
        self.u_lo = self.__apply_transform(self.u_lo, self.scalers[1])
        self.e_lo = self.__apply_transform(self.e_lo, self.scalers[1])

    def __apply_transform(self, x: np.ndarray, scaler: object) -> torch.Tensor:
        """Transform a contours according to a given scaler

        Args:
            x (np.ndarray): contours to be scaled
            scaler (object): fitted scaler

        Returns:
            torch.Tensor: fitted contours
        """

        out = scaler.transform(x.reshape(-1, 1)).squeeze(-1)
        return torch.from_numpy(out).float()

    def inverse_transform(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Inverse transform a vector (f0, lo)

        Args:
            x (torch.Tensor): contours vector

        Returns:
            list[np.ndarray]: [f0, lo] inverse transformed
        """

        f0, lo = torch.split(x, 1, -1)
        f0 = f0.reshape(-1, 1).cpu().numpy()
        lo = lo.reshape(-1, 1).cpu().numpy()

        # Inverse transforms
        f0 = self.scalers[0].inverse_transform(f0).reshape(-1)
        lo = self.scalers[1].inverse_transform(lo).reshape(-1)

        return [torch.Tensor(f0), torch.Tensor(lo)]

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

        return [u_contours, e_contours, s_onsets, s_offsets]
