import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle


class DatasetCreator:
    def __init__(self, path, filename):
        """Synthetic Dataset constructor. It aims at mimicking violin pitch and loudness contours

        """

        self.path = path
        self.filename = filename

        self.n_intervals = 6
        self.intervals = 1 / (np.power(2, np.arange(3, self.n_intervals)))
        self.vibrato_f = 1
        self.p_vibrato = 1.  # first we consider there is always a vibrato
        self.sparcity = .8

    def build(self, n: int, sample_length: int, type_lo: str) -> None:

        for _ in range(n):
            # initialize unexpressive contours:
            u_f0 = np.zeros(sample_length)
            u_lo = np.zeros(sample_length)

            # initialize expressive contours:
            e_f0 = np.zeros(sample_length)
            e_lo = np.zeros(sample_length)

            # initialize onsets and offsets
            onsets = np.zeros(sample_length)
            offsets = np.zeros(sample_length)

            # create list of masks
            mask = []

            start = 0

            while start < sample_length - 1:

                # include silent notes
                note_on = (np.random.random() < self.sparcity)
                # we build a new note:
                interval = np.random.choice(self.intervals)
                end = min(start + int(interval * sample_length), sample_length)

                if not note_on:
                    # update cursor
                    start = end
                else:
                    u_f0[start:end], e_f0[start:end] = self.__build_pitch(
                        start, end)
                    u_lo[start:end], e_lo[start:end] = self.__build_lo(
                        start, end, type=type_lo)

                    # update onsets and offsets:
                    onsets[start], offsets[min(end, sample_length - 1)] = 1, 1

                    # create mask and add it to the list
                    m = np.ones_like(onsets)
                    m[:start] -= 1
                    m[end:] -= 1
                    mask += [m]

                    # update cursor
                    start = end

            # export sample
            mask = np.array(mask)

            self.__export(u_f0, u_lo, e_f0, e_lo, onsets, offsets, mask)

    def __build_pitch(self, start: int, end: int) -> None:
        """Generate the pitch contours for one sample

        Args:
            start (int): start index of the sample
            length (int): length of the sample
        """

        # mimic frequency range of the violin (G3 -> 4 octaves)
        f = np.tile(np.random.randint(55, 55 + 4 * 12), end - start)

        # add some vibrato for expressive contours:
        v = .5 * np.sin(np.arange(end - start) / (self.vibrato_f))
        v = v * (np.random.random() < self.p_vibrato)
        # if hanning window
        v *= np.hanning(end - start)

        # add to contours:
        return f, (f + v)

    def __build_lo(self, start: int, end: int, type="peak") -> None:
        """Generate the loudness contours for one sample

        Args:
            start (int): start index of the sample
            length (int): length of the sample
            type (str, optional): MIDI loudness = type(Expressive loudness)
            peak and mean are implemented. Defaults to "peak".
        """

        length = end - start
        # mimic the amplitude range (MIDI norm [0, 128])
        amp = np.tile(np.random.randint(70, 128), length)

        #  hanning window
        # add expressive attack and release
        a1 = np.logspace(1,
                         np.log(amp[0] / 2),
                         length // 8 + 1,
                         base=np.exp(1))

        a2 = amp[:length // 8] - np.logspace(
            np.log(amp[0] / 2), 1, length // 8, base=np.exp(1))
        attack = np.concatenate([a1, a2])

        release = np.logspace(np.log(amp[0]), 1, length // 4, base=np.exp(1))
        e_lo = np.concatenate((attack, amp[:length // 2], release))

        if type == "mean":
            u_lo = np.tile(np.mean(e_lo), length)
        elif type == "peak":
            u_lo = np.tile(np.max(e_lo), length)
        else:
            raise ValueError

        return u_lo[:length], e_lo[:length]

    def __export(self, u_f0, u_lo, e_f0, e_lo, onsets, offsets, mask) -> None:
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

        with open(self.path + self.filename, "ab+") as file_out:
            pickle.dump(data, file_out)


if __name__ == '__main__':
    size = 100
    d = DatasetCreator("data/", f"dataset_synth_{size}.pickle")
    print("Build dataset")
    type_lo = "mean"
    d.build(100, 1024, type_lo)
