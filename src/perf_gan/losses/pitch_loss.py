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

    jit(nopython=False)

    def forward(self, gen_f0: List[torch.Tensor],
                sample: List[torch.Tensor]) -> torch.Tensor:
        # create the corresponding mask
        mask = self.__create_mask(sample)

        # get target f0
        t_f0, _ = sample[0].split(1, 1)

        t_f0.squeeze_(1)
        gen_f0.squeeze_(1)

        # apply mask to the pitch contours
        mk_gen_f0 = mask * gen_f0
        mk_t_f0 = mask * t_f0

        # compute the mean frequency for both contours
        mean_gen_f0 = torch.mean(mk_gen_f0,
                                 dim=-1) / (torch.mean(mask, dim=-1) + 1e-6)
        mean_t_f0 = torch.mean(mk_t_f0,
                               dim=-1) / (torch.mean(mask, dim=-1) + 1e-6)

        # compute the difference between mean frequency and loss
        diff = torch.abs(mean_gen_f0 - mean_t_f0)

        loss = torch.mean(torch.relu(diff - self.threshold))
        return loss

    jit(nopython=False)

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

        nb_sample, l = onsets.shape

        masks = []

        # we need to keep track of the max number of notes in a sample
        n_n_max = 0

        for i_sample in range(nb_sample):
            mask_sample = []

            #initialize current mask
            m = torch.zeros((1, l))

            for idx in range(l):
                if offsets[i_sample][idx]:
                    m[idx:] -= 1
                    # add mask to the sample mask
                    mask_sample += [m]
                    # reinitialize the mask
                    m = torch.zeros((1, l))

                if onsets[i_sample][idx]:
                    m[:idx] -= 1

            # update if necessary the highest number of notes in a sample
            n_n_max = max(n_n_max, len(mask_sample))
            masks += [torch.vstack(mask_sample)]

        # we need to pad the masks to the same size (n_n_max)
        for i in range(len(masks)):
            masks[i] = torch.nn.functional.pad(
                masks[i], (0, 0, 0, n_n_max - masks[i].shape[0]), value=0)

        return torch.stack(masks, dim=1)
