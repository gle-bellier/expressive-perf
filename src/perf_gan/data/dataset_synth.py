import numpy as np
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, sr=1600):
        self.e_f0 = None
        self.u_f0 = None
        self.samples_duration = None
        self.sr = sr

        self.n_intervals = 5
        self.intervals = 1 / (np.power(2, np.arange(self.n_intervals)))
        self.vibrato_f = 100
        self.p_vibrato = .25

    def build(self, n: int, duration: int):
        """Build the dataset composed of n samples of length duration (in s.)

        Args:
            n (int): number of samples in the dataset
            duration (int): duration (in s.) of each samples
        """
        self.samples_duration = duration

        self.e_f0 = np.zeros(n * duration * self.sr)
        self.u_f0 = np.zeros(n * duration * self.sr)
        start = 0

        while start < duration * n * self.sr:
            interval = np.random.choice(self.intervals)
            length = int(interval * self.sr)
            end = start + length
            # mimic frequency range of the violin (G3 -> 4 octaves)
            f = np.tile(np.random.randint(55, 55 + 4 * 12), length)

            # add some vibrato for expressive contours:
            v = .5 * np.sin(np.arange(length) / (self.sr / self.vibrato_f))
            v = v * (np.random.random() > self.p_vibrato)
            # if hanning window
            v *= np.hanning(length)

            # add to contours:
            self.e_f0[start:end] = (f + v)[:len(self.e_f0[start:end])]
            self.u_f0[start:end] = f[:len(self.u_f0[start:end])]
            start = end

    def show(self, n: int):
        """Show n samples of length duration * sr (in s.) of the dataset

        Args:
            n (int): number of samples to show
        """

        for i in range(n):
            plt.plot(self.u_f0[i * self.samples_duration * self.sr:(i + 1) *
                               self.samples_duration * self.sr],
                     label="unexpressive")
            plt.plot(self.e_f0[i * self.samples_duration * self.sr:(i + 1) *
                               self.samples_duration * self.sr],
                     label="expressive")
            plt.show()


if __name__ == '__main__':
    d = Dataset()
    d.build(10, 5)
    d.show(1)
