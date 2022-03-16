import pickle
from re import S
import torch
import torch.nn as nn


class Reverser:
    def __init__(self, path):
        self.path = path

        self.u_f0 = []
        self.u_lo = []
        self.onsets = []
        self.offsets = []
        self.mask = []
        self.__load()

    def __read_from_pickle(self, path):
        with open(path, 'rb') as file:
            try:
                while True:
                    yield pickle.load(file)

            except EOFError:
                pass

    def reverse(self):

        r_u_f0 = []
        r_u_lo = []
        r_onsets = []
        r_offsets = []
        r_mask = []

        print(len(self.u_f0))
        for i in range(len(self.u_f0)):
            r_u_f0 += [self.u_f0[i][::-1]]
            r_u_lo += [self.u_lo[i][::-1]]
            r_onsets += [self.onsets[i][::-1]]
            r_offsets += [self.offsets[i][::-1]]
            r_mask += [self.mask[i][..., ::-1]]

        self.u_f0 += r_u_f0
        self.u_lo += r_u_lo
        self.onsets += r_onsets
        self.offsets += r_offsets
        self.mask += r_mask

    def __load(self):

        for c in self.__read_from_pickle(self.path):

            self.u_f0 += [c["f0"]]
            self.u_lo += [c["lo"]]
            self.onsets += [c["onsets"]]
            self.offsets += [c["offsets"]]
            self.mask += [c["mask"]]

    def write(self, path):

        print("u_f0 length: ", len(self.u_f0))
        print("u_lo length: ", len(self.u_lo))
        print("onsets: ", len(self.onsets))
        print("offsets: ", len(self.offsets))
        print("mask length: ", len(self.mask))

        for i in range(len(self.u_f0)):
            self.__export(path, self.u_f0[i], self.u_lo[i], self.onsets[i],
                          self.offsets[i], self.mask[i])

    def __export(self, path, u_f0, u_lo, onsets, offsets, mask) -> None:
        """Exporting the dataset to a pickle file 

        Args:
            path (str): Path to the directory
            filename (str): exported file name
        """
        data = {
            "f0": u_f0,
            "lo": u_lo,
            "onsets": onsets,
            "offsets": offsets,
            "mask": mask
        }

        with open(path, "ab+") as file_out:
            pickle.dump(data, file_out)


def main():
    path = "data/midi/contours/midi_contours.pickle"
    saving_path = "data/midi/contours/midi_contours_r.pickle"
    da = Reverser(path)
    da.reverse()
    da.write(saving_path)
    print(len(da.u_lo))


if __name__ == "__main__":
    main()