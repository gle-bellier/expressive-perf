import numpy as np
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, sr=1600):

        self.u_f0 = None
        self.u_lo = None
        self.e_f0 = None
        self.e_l0 = None

        self.samples_duration = None
        self.sr = sr

        self.n_intervals = 5
        self.intervals = 1 / (np.power(2, np.arange(self.n_intervals)))
        self.vibrato_f = 100
        self.p_vibrato = 1.  # first we consider there is always a vibrato
        self.sparcity = .5

    def build(self, n: int, duration: int):
        """Build the dataset composed of n samples of length duration (in s.)

        Args:
            n (int): number of samples in the dataset
            duration (int): duration (in s.) of each samples
        """
        self.samples_duration = duration

        self.u_f0 = np.zeros(n * duration * self.sr)
        self.u_lo = np.zeros(n * duration * self.sr)

        self.e_f0 = np.zeros(n * duration * self.sr)
        self.e_lo = np.zeros(n * duration * self.sr)
        start = 0

        while start < duration * n * self.sr:
            interval = np.random.choice(self.intervals)
            length = int(interval * self.sr)
            self._build_pitch(start, length)
            self._build_lo(start, length)
            start += length

    def _build_pitch(self, start: int, length: int):
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

    def _build_lo(self, start: int, length: int):
        end = start + length
        # mimic the amplitude range (MIDI norm [0, 255])
        amp = np.tile(np.random.randint(180, 230), length)
        # include silent notes
        amp *= (np.random.random() < self.sparcity)
        #  hanning window
        window = np.hanning(length)

        # add to contours:
        self.u_lo[start:end] = amp[:len(self.e_lo[start:end])]
        self.e_lo[start:end] = (amp * window)[:len(self.u_lo[start:end])]

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

            plt.plot(self.u_lo[i * self.samples_duration * self.sr:(i + 1) *
                               self.samples_duration * self.sr],
                     label="unexpressive")
            plt.plot(self.e_lo[i * self.samples_duration * self.sr:(i + 1) *
                               self.samples_duration * self.sr],
                     label="expressive")
            plt.show()


if __name__ == '__main__':
    d = Dataset()
    d.build(10, 5)
    d.show(1)
