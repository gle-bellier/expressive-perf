import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle
import numpy as np
from typing import List, Union
from random import randint
import re

from perf_gan.data.preprocess import Identity, PitchTransform, LoudnessTransform


class ContoursDataset(Dataset):
    def __init__(self, path: str, list_transforms=None, valid=False):
        self.path = path
        self.list_transforms = list_transforms
        self.valid = valid

        self.u_f0 = []
        self.e_f0 = []
        self.u_lo = []
        self.e_lo = []
        self.onsets = []
        self.offsets = []
        self.mask = []

        print("Loading dataset.")
        self.__load()
        print(f"Dataset of {len(self.u_f0)} samples loaded")

        # we need to convert Hz -> MIDI for e_f0
        for i in range(len(self.e_f0)):
            self.e_f0[i] = self.__hz2midi()

        # we need to convert MIDI -> dB for u_lo
        # And not the other way since we need to not compute
        # any non differentiable inverse transformation before using
        # ddsp since we have a spectrogram loss on reconstruction

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
            self.u_f0 += [torch.Tensor(c["u_f0"])]
            self.e_f0 += [torch.Tensor(c["e_f0"])]
            self.u_lo += [torch.Tensor(c["u_lo"])]
            self.e_lo += [torch.Tensor(c["e_lo"])]
            self.onsets += [torch.Tensor(c["onsets"])]
            self.offsets += [torch.Tensor(c["offsets"])]
            self.mask += [torch.Tensor(c["mask"])]


def main():

    d = ContoursDataset("data/dataset.pickle")


if __name__ == "__main__":
    main()