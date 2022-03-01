import pickle
from re import S
import torch
import torch.nn as nn


class DataAugmentation:

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

    def f0_shift(self, shifts):

        for shift in shifts:

            assert int(
                shift) == shift, f"Shift should be an Integer, got {shift}"

            for i in range(len(self.u_f0)):
                # apply shift on pitches (midi norm)
                self.u_f0 += [self.u_f0[i] + shift]
                self.e_f0 += [self.e_f0[i] + shift]

            # replicate others

            self.u_lo += self.u_lo
            self.e_lo += self.e_lo
            self.onsets += self.onsets
            self.offsets += self.offsets
            self.mask += self.mask

    def __load(self):

        for c in self.__read_from_pickle(self.path):

            self.u_f0 += [c["u_f0"]]

            self.e_f0 += [c["e_f0"]]
            self.u_lo += [c["u_lo"]]
            self.e_lo += [c["e_lo"]]
            self.onsets += [c["onsets"]]
            self.offsets += [c["offsets"]]
            self.mask += [c["mask"]]

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
    path = "data/dataset.pickle"
    saving_path = "data/dataset_aug.pickle"
    da = DataAugmentation(path)
    da.f0_shift([-2, -1, 1, 2])

    da.write(saving_path)
    print(len(da.u_lo))


if __name__ == "__main__":
    main()