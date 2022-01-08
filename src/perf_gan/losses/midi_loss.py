import torch
import torch.nn as nn
from typing import List
from numba import jit


class Midi_loss(nn.Module):
    def __init__(self, f0_threshold=.5, lo_threshold=.5):
        """Pitch loss estimating how accurate are expressive pitch contours according to 
        the unexpressive contours (the reference). Note to note mean frequency comparison. 

        Args:
            threshold (float): threshold above which the contours is considered off (in the midi norm, 0.5 is a quarter tone)
        """
        super(Midi_loss, self).__init__()
        self.f0_threshold = f0_threshold
        self.lo_threshold = lo_threshold

    jit(nopython=True, parallel=True)

    def forward(self,
                gen_f0,
                t_f0,
                gen_lo,
                t_lo,
                onsets,
                offsets,
                types=["mean", "mean"]) -> torch.Tensor:
        # create the corresponding mask
        mask = self.__create_mask(onsets, offsets).to(gen_f0.device)

        # apply mask to the pitch contours
        mk_gen_f0 = mask * gen_f0.squeeze(1)
        mk_t_f0 = mask * t_f0.squeeze(1)

        # apply mask to the loudness contours
        mk_gen_lo = mask * gen_lo.squeeze(1)
        mk_t_lo = mask * t_lo.squeeze(1)

        loss_pitch = self.__contour_loss(mk_gen_f0, mk_t_f0, mask,
                                         self.f0_threshold, types[0])

        loss_lo = self.__contour_loss(mk_gen_lo, mk_t_lo, mask,
                                      self.lo_threshold, types[1])

        return loss_pitch, loss_lo

    jit(nopython=True, parallel=True)

    def __contour_loss(self, mk_gen, mk_target, mask, threshold, alg):
        loss_mean = loss_prop = 0

        if alg in ["mean", "both"]:
            # compute the means for each notes for both contours
            mean_gen = torch.mean(mk_gen,
                                  dim=-1) / (torch.mean(mask, dim=-1) + 1e-6)
            mean_target = torch.mean(
                mk_target, dim=-1) / (torch.mean(mask, dim=-1) + 1e-6)

            # compute the difference between means
            diff = torch.abs(mean_gen - mean_target)

            loss_mean = torch.mean(torch.relu(diff - threshold))

        if alg != "mean":
            # we compute a proportional loss
            diff = torch.abs(mk_gen - mk_target)
            loss_prop = torch.sum((diff > threshold).float()) / torch.sum(mask)

        return (loss_mean + loss_prop) * (1 - 0.5 * (alg == "both"))

    jit(nopython=True, parallel=True)

    def __create_mask(self, onsets, offsets) -> torch.Tensor:
        """Create a temporal mask according to notes onsets and offsets.
        Each column of the mask correspond to the temporal activation of a
        single note
     

        Args:
            x (list[torch.Tensor]): sample (unexpressive,  expressive contours, onsets, offsets)

        Returns:
            torch.Tensor: mask of size (len(onsets) x number of notes)
        """

        nb_sample, l = onsets.shape

        masks = []

        # we need to keep track of the max number of notes in a sample
        n_n_max = 0

        for i_sample in range(nb_sample):
            mask_sample = []

            #initialize current mask
            m = torch.ones((1, l))

            for idx in range(l):
                if offsets[i_sample][idx]:
                    m[:, idx:] -= 1
                    # add mask to the sample mask
                    mask_sample += [m]

                    # reinitialize the mask
                    m = torch.ones((1, l))

                if onsets[i_sample][idx]:
                    m[:, :idx] -= 1

            # update if necessary the highest number of notes in a sample
            n_n_max = max(n_n_max, len(mask_sample))
            masks += [torch.vstack(mask_sample)]

        # we need to pad the masks to the same size (n_n_max)
        for i in range(len(masks)):
            masks[i] = torch.nn.functional.pad(
                masks[i], (0, 0, 0, n_n_max - masks[i].shape[0]), value=0)

        return torch.stack(masks, dim=1)
