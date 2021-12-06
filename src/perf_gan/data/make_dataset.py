import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle


class DatasetCreator:
    def __init__(self, sr=1600):
        """Synthetic Dataset constructor. It aims at mimicking violin pitch and loudness contours

        Args:
            sr (int, optional): sampling rate. Defaults to 1600.
        """

        # unexpressive contours:
        self.u_f0 = None
        self.u_lo = None
        # expressive contours:
        self.e_f0 = None
        self.e_l0 = None
        # onsets and offsets:
        self.onsets = None
        self.offsets = None

        self.samples_duration = None
        self.sr = sr

        self.n_intervals = 5
        self.intervals = 1 / (np.power(2, np.arange(self.n_intervals)))
        self.vibrato_f = 100
        self.p_vibrato = 1.  # first we consider there is always a vibrato
        self.sparcity = .8

    def build(self, n: int, duration: int) -> None:
        """Build the dataset composed of n samples of length duration (in s.)

        Args:
            n (int): number of samples in the dataset
            duration (int): duration (in s.) of each samples
        """
        self.samples_duration = duration

        # initialize unexpressive contours:
        self.u_f0 = np.zeros(n * duration * self.sr)
        self.u_lo = np.zeros(n * duration * self.sr)

        # initialize expressive contours:
        self.e_f0 = np.zeros(n * duration * self.sr)
        self.e_lo = np.zeros(n * duration * self.sr)

        # initialize onsets and offsets
        self.onsets = np.zeros(n * duration * self.sr)
        self.offsets = np.zeros(n * duration * self.sr)

        start = 0

        while start < duration * n * self.sr:
            interval = np.random.choice(self.intervals)
            length = int(interval * self.sr)
            self.__build_pitch(start, length)
            self.__build_lo(start, length, type="mean")
            start += length

    def __build_pitch(self, start: int, length: int) -> None:
        """Generate the pitch contours for one sample

        Args:
            start (int): start index of the sample
            length (int): length of the sample
        """
        end = start + length

        # mimic frequency range of the violin (G3 -> 4 octaves)
        f = np.tile(np.random.randint(55, 55 + 4 * 12), length)

        # add some vibrato for expressive contours:
        v = .5 * np.sin(np.arange(length) / (self.sr / self.vibrato_f))
        v = v * (np.random.random() < self.p_vibrato)
        # if hanning window
        v *= np.hanning(length)

        # add to contours:
        self.e_f0[start:end] = (f + v)[:len(self.e_f0[start:end])]
        self.u_f0[start:end] = f[:len(self.u_f0[start:end])]

    def __build_lo(self, start: int, length: int, type="peak") -> None:
        """Generate the loudness contours for one sample

        Args:
            start (int): start index of the sample
            length (int): length of the sample
            type (str, optional): MIDI loudness = type(Expressive loudness)
            peak and mean are implemented. Defaults to "peak".
        """

        end = start + length
        # mimic the amplitude range (MIDI norm [0, 255])
        amp = np.tile(np.random.randint(100, 230), length)

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

        # include silent notes
        note_on = (np.random.random() < self.sparcity)

        # create contours:
        self.e_lo[start:end] = e_lo[:len(self.e_lo[start:end])] * note_on

        if type == "mean":
            self.u_lo[start:end] = np.tile(np.mean(e_lo * note_on),
                                           len(self.e_lo[start:end]))
        elif type == "peak":
            self.u_lo[start:end] = np.tile(np.max(e_lo * note_on),
                                           len(self.e_lo[start:end]))
        else:
            raise ValueError

        # update onsets and offsets:
        if note_on:
            self.onsets[start] = 1
            if end < len(self.offsets):
                self.offsets[end] = 1

    def export(self, path: str, filename: str) -> None:
        """Exporting the dataset to a pickle file 

        Args:
            path (str): Path to the directory
            filename (str): exported file name
        """
        data = {
            "u_f0": self.u_f0,
            "u_lo": self.u_lo,
            "e_f0": self.e_f0,
            "e_lo": self.e_lo,
            "onsets": self.onsets,
            "offsets": self.offsets
        }

        with open(path + filename, "wb") as file_out:
            pickle.dump(data, file_out)


if __name__ == '__main__':
    d = DatasetCreator()
    d.build(1000, 5)
    d.export("data/", "dataset.pickle")
