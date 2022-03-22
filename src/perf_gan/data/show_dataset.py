import pickle
import matplotlib.pyplot as plt


class Visualizer:

    def __init__(self, path):
        self.path = path

    def show(self, n: int) -> None:
        """Show the n first samples of the dataset

        Args:
            n (int): index of the sample to show
        """
        with open(self.path, "rb") as dataset_file:
            for i in range(n):
                dataset = pickle.load(dataset_file)
                u_f0, u_lo = dataset["u_f0"], dataset["u_lo"]
                e_f0, e_lo = dataset["e_f0"], dataset["e_lo"]
                onsets, offsets = dataset["onsets"], dataset["offsets"]
                mask = dataset["mask"]

                plt.subplot(1, 3, 1)
                plt.plot(u_f0, label="unexpressive", color="red")
                plt.plot(e_f0, label="expressive", color="blue")
                plt.plot(onsets, label="onsets")
                plt.plot(offsets, label="offsets")
                plt.xlabel("time")
                plt.ylabel("p")
                plt.legend()
                plt.title("Pitch (MIDI)")

                plt.subplot(1, 3, 2)
                plt.plot(u_lo, label="unexpressive", color="red")
                plt.plot(e_lo, label="expressive", color="blue")
                plt.plot(onsets, label="onsets")
                plt.plot(offsets, label="offsets")
                plt.xlabel("time")
                plt.ylabel("V")
                plt.legend()
                plt.title("Volume (MIDI)")
                plt.legend()
                plt.plot()

                plt.figure(figsize=(5, 8))
                plt.matshow(mask, fignum=1, aspect='auto')
                plt.title("mask")
                plt.show()


if __name__ == '__main__':
    v = Visualizer("data/train_c.pickle")

    v.show(10)