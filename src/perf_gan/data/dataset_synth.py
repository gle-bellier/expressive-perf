import numpy as np
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, n):
        self.n = n

    def show(self, n, duration=1):
        """Show n samples of length duration * sr (in s.) of the dataset

        Args:
            n ([type]): number of samples to show
            duration (int, optional): duration of each samples to show. Defaults to 1.
        """
        pass