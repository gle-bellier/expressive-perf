import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns

from perf_gan.data.dataset import GANDataset


def create_mask(x: list[torch.Tensor]) -> torch.Tensor:
    onsets, offsets = x[-2:]
    masks = []
    # initialize a mask
    m = torch.ones_like(onsets)
    for idx in range(len(onsets)):
        if offsets[idx]:
            m[idx:] -= 1
            # add mask to the mask
            masks += [m]
            # reinitialize the mask
            m = torch.ones_like(onsets)

        if onsets[idx]:
            m[:idx] -= 1

    return torch.stack(masks).T


if __name__ == '__main__':
    transforms = [(StandardScaler, {}), (MinMaxScaler, {})]
    dataset = GANDataset("data/dataset.pickle", 160 * 5, transforms)
    dataset.transform()
    x = dataset[0]
    mask = create_mask(x)

    e_f0, e_lo = dataset.inverse_transform(x[0])
    e_f0.unsqueeze_(-1)
    e_lo.unsqueeze_(-1)
    masked_e_f0 = e_f0 * mask
    masked_e_lo = e_lo * mask

    for i in range(mask.shape[1]):
        plt.subplot(1, 2, 1)
        plt.plot(e_f0)
        plt.plot(masked_e_f0[:, i])
        plt.subplot(1, 2, 2)
        plt.plot(e_lo)
        plt.plot(masked_e_lo[:, i])
        plt.show()
