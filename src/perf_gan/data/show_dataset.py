import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from perf_gan.data.dataset import GANDataset


class Visualizer:
    def __init__(self, path):

        transforms = [(StandardScaler, {}), (MinMaxScaler, {})]
        self.dataset = GANDataset(path, 1600 * 5, transforms)
        self.dataset.transform()

    def show(self, n: int) -> None:
        """Show the nth sample of the dataset

        Args:
            n (int): index of the sample to show
        """
        sample = self.dataset[n]
        u_f0, u_lo = torch.split(sample[0], 1, -1)
        e_f0, e_lo = torch.split(sample[1], 1, -1)
        onsets, offsets = sample[2], sample[3]

        plt.subplot(1, 2, 1)
        plt.plot(u_f0, label="unexpressive")
        plt.plot(e_f0, label="expressive")
        plt.plot(onsets, label="onsets")
        plt.plot(offsets, label="offsets")
        plt.title("pitch")
        plt.subplot(1, 2, 2)
        plt.plot(u_lo, label="unexpressive")
        plt.plot(e_lo, label="expressive")
        plt.plot(onsets, label="onsets")
        plt.plot(offsets, label="offsets")
        plt.title("loudness")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    v = Visualizer("data/dataset.pickle")
    for i in range(4):
        v.show(i)