from perf_gan.data.dataset_synth import Dataset


class Model:
    def __init__(self):
        pass

    def forward(self, x):
        return x**2


if __name__ == '__main__':
    d = Dataset(3)
    print(d.get())