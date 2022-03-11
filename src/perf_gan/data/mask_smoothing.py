import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# t = np.arange(1, 1000, 1)

# win = signal.windows.hann(100)
# s = np.zeros((2, 1024))
# s[0, 400:600] += 1
# s[0, 800:900] += 1

# sm_mask = []

# for note in s:
#     sm = (1 - signal.convolve(1 - note, win, mode='same') / sum(win)) * note
#     sm_mask += [sm]

# plt.plot(s[0])
# plt.plot(s[1])
# plt.plot(sm_mask[0], label="0")
# plt.plot(sm_mask[1], label="1")
# plt.legend()
# plt.show()

import pickle
from re import S
import torch
import torch.nn as nn


class MaskSmoothing:

    def __init__(self, path):
        self.path = path

        self.u_f0 = []
        self.e_f0 = []
        self.u_lo = []
        self.e_lo = []
        self.onsets = []
        self.offsets = []
        self.mask = []
        self.__load()

        print(len(self.u_f0))

    def __read_from_pickle(self, path):
        with open(path, 'rb') as file:
            try:
                while True:
                    yield pickle.load(file)

            except EOFError:
                pass

    def smoothing(self, width):
        win = signal.windows.hann(width)

        list_mask = []
        for mask in self.mask:
            smooth = []
            for note in mask:
                note = note.detach().numpy()
                sm = (1 - signal.convolve(1 - note, win, mode='same') /
                      sum(win)) * note
                smooth += [sm]

            list_mask += [torch.tensor(smooth)]

        self.mask = list_mask

    def __load(self):

        for c in self.__read_from_pickle(self.path):

            self.u_f0 += [torch.tensor(c["u_f0"]).float()]

            self.e_f0 += [torch.tensor(c["e_f0"]).float()]
            self.u_lo += [torch.tensor(c["u_lo"]).float()]
            self.e_lo += [torch.tensor(c["e_lo"]).float()]
            self.onsets += [torch.tensor(c["onsets"]).float()]
            self.offsets += [torch.tensor(c["offsets"]).float()]
            self.mask += [torch.tensor(c["mask"]).float()]

    def write(self, path):

        print("u_f0 length: ", len(self.u_f0))
        print("u_lo length: ", len(self.u_lo))
        print("e_f0 length: ", len(self.e_f0))
        print("e_lo length: ", len(self.e_lo))
        print("onsets: ", len(self.onsets))
        print("offsets: ", len(self.offsets))
        print("mask length: ", len(self.mask))

        for i in range(len(self.u_f0)):
            self.__export(path, self.u_f0[i], self.u_lo[i], self.e_f0[i],
                          self.e_lo[i], self.onsets[i], self.offsets[i],
                          self.mask[i])

    def __export(self, path, u_f0, u_lo, e_f0, e_lo, onsets, offsets,
                 mask) -> None:
        """Exporting the dataset to a pickle file 

        Args:
            path (str): Path to the directory
            filename (str): exported file name
        """
        data = {
            "u_f0": u_f0,
            "u_lo": u_lo,
            "e_f0": e_f0,
            "e_lo": e_lo,
            "onsets": onsets,
            "offsets": offsets,
            "mask": mask
        }

        with open(path, "ab+") as file_out:
            pickle.dump(data, file_out)


def main():
    path = "data/dataset_aug.pickle"
    saving_path = "data/dataset_aug_ms.pickle"
    ms = MaskSmoothing(path)
    ms.smoothing(64)

    ms.write(saving_path)


if __name__ == "__main__":
    main()