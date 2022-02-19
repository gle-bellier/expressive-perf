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
                mask,
                types=["mean", "mean"],
                abs=[True, True]) -> torch.Tensor:

        # apply mask to the pitch contours

        mk_gen_f0 = mask * gen_f0
        mk_t_f0 = mask * t_f0

        # apply mask to the loudness contours
        mk_gen_lo = mask * gen_lo
        mk_t_lo = mask * t_lo

        loss_pitch = self.__contour_loss(mk_gen_f0,
                                         mk_t_f0,
                                         mask,
                                         self.f0_threshold,
                                         types[0],
                                         abs=abs[0])

        loss_lo = self.__contour_loss(mk_gen_lo,
                                      mk_t_lo,
                                      mask,
                                      self.lo_threshold,
                                      types[1],
                                      abs=abs[0])

        return loss_pitch, loss_lo

    jit(nopython=True, parallel=True)

    def __contour_loss(self,
                       mk_gen,
                       mk_target,
                       mask,
                       threshold,
                       alg,
                       abs=True):
        loss_mean = loss_prop = 0

        if alg in ["mean", "both"]:
            # compute the means for each notes for both contours
            mean_gen = torch.mean(mk_gen,
                                  dim=-1) / (torch.mean(mask, dim=-1) + 1e-6)
            mean_target = torch.mean(
                mk_target, dim=-1) / (torch.mean(mask, dim=-1) + 1e-6)

            # compute the difference between means
            if abs:
                diff = torch.abs(mean_gen - mean_target)
            else:
                diff = torch.abs(mean_gen - mean_target)

            loss_mean = torch.mean(torch.relu(diff - threshold))

        if alg != "mean":
            # we compute a proportional loss
            if abs:
                diff = torch.abs(mk_gen - mk_target)
            else:
                mk_gen - mk_target
            loss_prop = torch.sum((diff > threshold).float()) / torch.sum(mask)

        return (loss_mean + loss_prop) * (1 - 0.5 * (alg == "both"))

    jit(nopython=True, parallel=True)
