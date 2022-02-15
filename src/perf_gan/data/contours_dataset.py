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

        contour = np.concatenate((np.concatenate(u_f0), np.concatenate(e_f0)))

        transform = self.list_transforms[0]
        sc = transform[0](**transform[1]).fit(contour.reshape(-1, 1))
        scalers.append(sc)

        # loudness
        contour = np.concatenate((np.concatenate(u_lo), np.concatenate(e_lo)))
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
        for c in self.__read_from_pickle(self.path):
            self.u_f0 += [torch.tensor(c["u_f0"])]
            self.e_f0 += [torch.tensor(c["e_f0"])]
            self.u_lo += [torch.tensor(c["u_lo"])]
            self.e_lo += [torch.tensor(c["e_lo"])]
            self.onsets += [torch.tensor(c["onsets"])]
            self.offsets += [torch.tensor(c["offsets"])]
            self.mask += [torch.tensor(c["mask"])]

            self.length += 1

    def __getitem__(self, index) -> torch.Tensor:

        # add some jit in the case of training dataset
        if not self.valid:
            index += int(torch.randint(-5, 5, (1, )))

        # check index still in range
        index = max(0, min(index, self.length - 1))

        # scaling item contours:

        u_f0, u_lo = self.transform(self.u_f0[index], self.u_lo[index])
        e_f0, e_lo = self.transform(self.e_f0[index], self.e_lo[index])

        u_c = torch.cat([u_f0, u_lo], 0)
        e_c = torch.cat([e_f0, e_lo], 0)

        # get onsets, offsets, mask

        onsets = self.onsets[index]
        offsets = self.offsets[index]
        mask = self.mask[index]

        return u_c, e_c, onsets, offsets, mask


def main():

    import matplotlib.pyplot as plt

    l = [(PitchTransform, {
        "feature_range": (-1, 1)
    }), (LoudnessTransform, {
        "feature_range": (-1, 1)
    })]
    d = ContoursDataset("data/dataset.pickle", l)

    u_c, e_c, onsets, offsets, mask = d[0]

    print(type(u_c))

    print(type(onsets))

    print(type(mask))


if __name__ == "__main__":
    main()