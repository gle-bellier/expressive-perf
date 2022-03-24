import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle
import numpy as np
from typing import List, Union, Tuple
from random import randint
import re

from perf_gan.data.preprocess import Identity, PitchTransform, LoudnessTransform


class ContoursDataset(Dataset):

    def __init__(self, path: str, list_transforms=None, valid=False):
        self.path = path

        self.valid = valid

        self.u_f0 = []
        self.e_f0 = []
        self.u_lo = []
        self.e_lo = []
        self.onsets = []
        self.offsets = []
        self.mask = []

        self.length = 0

        print("Loading dataset.")
        self.__load()
        print(f"Dataset of {len(self.u_f0)} samples loaded")

        # fitting scalers to the distributions

        if list_transforms is None:
            self.list_transforms = [(Identity, {}), (Identity, {})]
        else:
            self.list_transforms = list_transforms

        self.scalers = self.__fit_transforms(self.u_f0, self.u_lo, self.e_f0,
                                             self.e_lo)

    def __len__(self):
        return self.length

    def __fit_transforms(self, u_f0, u_lo, e_f0, e_lo) -> List[object]:
        """Fit the two transforms to the contours

        Returns:
            list[object]: fitted scalers
        """
        scalers = []

        # pitch

        contour = torch.cat([torch.cat(u_f0), torch.cat(e_f0)])

        transform = self.list_transforms[0]
        sc = transform[0](**transform[1]).fit(contour.reshape(-1, 1))
        scalers.append(sc)

        # loudness
        contour = torch.cat([torch.cat(u_lo), torch.cat(e_lo)])
        transform = self.list_transforms[1]
        sc = transform[0](**transform[1]).fit(contour.reshape(-1, 1))
        scalers.append(sc)

        return scalers

    def transform(self, f0, lo) -> Tuple[torch.Tensor]:
        """Apply transformations to the contour x (pitch and loudness)

        Args:
            
            f0 (torch.Tensor): fundamental frequency
            lo (torch.Tensor): loudness

        Returns:
            torch.Tensor: f0 : fundamental frequency
            torch.Tensor: lo : loudness
            
        """

        # transforms
        f0 = self.scalers[0].transform(f0)
        lo = self.scalers[1].transform(lo)

        return f0, lo

    def inverse_transform(self, f0: torch.Tensor,
                          lo: torch.Tensor) -> torch.Tensor:
        """Apply inverse transformations to the contour x (pitch and loudness)

        Args:
            x (torch.Tensor): input contour

        Returns:
            torch.Tensor: inverse transformed contour
        """

        # Inverse transforms
        f0 = self.scalers[0].inverse_transform(f0)
        lo = self.scalers[1].inverse_transform(lo)

        return f0, lo

    def __hz2midi(self, f):
        return 12 * torch.log(f / 440) + 69

    def __midi2hz(self, x):
        return torch.pow(2, (x - 69) / 12) * 440

    def __read_from_pickle(self, path):
        with open(path, 'rb') as file:
            try:
                while True:
                    yield pickle.load(file)
            except EOFError:
                pass

    def __load(self):
        # load all contours

        max_mask_size = 0
        l_mask = []

        for c in self.__read_from_pickle(self.path):
            u_f0 = torch.tensor(c["u_f0"]).float()
            e_f0 = torch.tensor(c["e_f0"]).float()
            u_lo = torch.tensor(c["u_lo"]).float()
            e_lo = torch.tensor(c["e_lo"]).float()
            onsets = torch.tensor(c["onsets"]).float()
            offsets = torch.tensor(c["offsets"]).float()

            self.u_f0 += [u_f0[:512], u_f0[512:]]
            self.e_f0 += [e_f0[:512], e_f0[512:]]
            self.u_lo += [u_lo[:512], u_lo[512:]]
            self.e_lo += [e_lo[:512], e_lo[512:]]
            self.onsets += [onsets[:512], onsets[512:]]
            self.offsets += [offsets[:512], offsets[512:]]

            # we need to keep track of the "largest" mask for future padding

            m = torch.tensor(c["mask"]).float()
            max_mask_size = max(max_mask_size, m.shape[0])
            l_mask += [m]

            self.length += 1

        # pad all mask to the maximum size
        for m in l_mask:
            pd = (0, 0, 0, max_mask_size - m.shape[0])
            m = torch.nn.functional.pad(m, pd)
            self.mask += [m[:, :512], m[:, 512:]]

        # mask padding

    def __getitem__(self, index) -> torch.Tensor:

        # add some jit in the case of training dataset
        if not self.valid:
            index += int(torch.randint(-5, 5, (1, )))

        # check index still in range
        index = max(0, min(index, self.length - 1))

        # scaling item contours:

        u_f0, u_lo = self.transform(self.u_f0[index], self.u_lo[index])
        e_f0, e_lo = self.transform(self.e_f0[index], self.e_lo[index])

        # reshape vectors

        u_f0 = u_f0.unsqueeze(0)
        u_lo = u_lo.unsqueeze(0)
        e_f0 = e_f0.unsqueeze(0)
        e_lo = e_lo.unsqueeze(0)

        # get onsets, offsets, mask

        onsets = self.onsets[index].unsqueeze(0)
        offsets = self.offsets[index].unsqueeze(0)
        mask = self.mask[index]

        return u_f0, u_lo, e_f0, e_lo, onsets, offsets, mask
