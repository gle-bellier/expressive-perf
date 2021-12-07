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
    masked_e_f0 = mask * e_f0
    masked_e_lo = mask * e_lo

    u_f0, u_lo = dataset.inverse_transform(x[1])
    u_f0.unsqueeze_(-1)
    u_lo.unsqueeze_(-1)
    masked_u_f0 = mask * u_f0
    masked_u_lo = mask * u_lo

    # plt.subplot(1, 2, 1)
    # plt.plot(masked_u_f0)
    # plt.plot(masked_e_f0)
    # plt.subplot(1, 2, 2)
    # plt.plot(masked_u_lo)
    # plt.plot(masked_e_lo)
    # plt.show()

    # plt.subplot(1, 2, 1)
    # plt.plot(masked_u_f0 - masked_e_f0)
    # plt.subplot(1, 2, 2)
    # plt.plot(masked_u_lo - masked_e_lo)
    # plt.show()

    diff = torch.abs(masked_e_f0 - masked_u_f0)
    print((diff > .4).shape)
    loss = torch.mean((diff > .4).float()) / torch.mean(mask)
    print(loss)