import numpy as np
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, n, sr=1600):
        self.n = None
        self.contours = None
        self.sr = sr

    def build(self, n: int, duration: int):
        """Build the dataset composed of n samples of length duration (in s.)

        Args:
            n (int): number of samples in the dataset
            duration (int): duration (in s.) of each samples
        """

        self.contours = np.random.random(n * duration * self.sr)

    def show(self, n: int, duration=1):
        """Show n samples of length duration * sr (in s.) of the dataset

        Args:
            n (int): number of samples to show
            duration (int, optional): duration of each samples to show. Defaults to 1.
        """

        for i in range(n):
            plt.plot(self.contours[n * duration * self.sr:(n + 1) * duration *
                                   self.sr])
            plt.show()
