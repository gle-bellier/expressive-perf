import torch
import torch.nn as nn
from typing import List
from numba import jit


class PitchLoss(nn.Module):
    def __init__(self, threshold=.5):
        """Pitch loss estimating how accurate are expressive pitch contours according to 
        the unexpressive contours (the reference). Note to note mean frequency comparison. 

        Args:
            threshold (float): threshold above which the contours is considered off (in the midi norm, 0.5 is a quarter tone)
        """
        super(PitchLoss, self).__init__()
        self.threshold = threshold

    def forward(self, gen_f0: List[torch.Tensor],
                sample: List[torch.Tensor]) -> torch.Tensor:
        # create the corresponding mask
        mask = self.__create_mask(sample).unsqueeze(0)

        # get target f0
        t_f0, _ = sample[0].split(1, 1)

        t_f0.squeeze_(1)
        gen_f0.squeeze_(1)

        print("t_f0 : ", t_f0.shape)

        print("gen_f0 : ", gen_f0.shape)

        print("mask : ", mask.shape)

        # apply mask to the pitch contours
        mk_gen_f0 = mask * gen_f0
        mk_t_f0 = mask * t_f0

        # compute the mean frequency for both contours
        mean_gen_f0 = torch.mean(mk_gen_f0, dim=0) / torch.mean(mask, dim=0)
        mean_t_f0 = torch.mean(mk_t_f0, dim=0) / torch.mean(mask, dim=0)

        # compute the difference between mean frequency and loss
        diff = torch.abs(mean_gen_f0 - mean_t_f0)
        loss = torch.mean((diff > self.threshold).float())

        return loss

    def __create_mask(self, x: List[torch.Tensor]) -> torch.Tensor:
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
        print("Mask shape : ", m.shape)
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
