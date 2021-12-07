import torch
import torch.nn as nn


class PitchLoss(nn.Module):
    def __init__(self, threshold: .5):
        super().__init(self)
        self.threshold = threshold

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:

        # create the corresponding mask
        mask = self.__create_mask(x)

        u_f0, _ = x[0].split(1, -1)
        e_f0, _ = x[0].split(1, -1)

        # apply mask to the pitch contours
        mk_u_f0 = mask * u_f0
        mk_e_f0 = mask * e_f0

        # compute the midi pitch differences
        diff = torch.abs(mk_e_f0 - mk_u_f0)
        # proportion of contours points where pitch is wrong (diff above threshold)
        loss = torch.mean((diff > self.threshold).float()) / torch.mean(mask)

        return loss

    def __create_mask(x: list[torch.Tensor]) -> torch.Tensor:
        """Create a temporal mask according to notes onsets and offsets.
        Each column of the mask correspond to the temporal activation of a
        single note
     

        Args:
            x (list[torch.Tensor]): sample (unexpressive,  expressive contours, onsets, offsets)

        Returns:
            torch.Tensor: mask of size (len(onsets) x number of notes)
        """
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