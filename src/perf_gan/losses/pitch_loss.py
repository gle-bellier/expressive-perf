import torch
import torch.nn as nn
from typing import List
from numba import jit


class PitchLoss(nn.Module):
    def __init__(self, threshold: .5):
        """Pitch loss estimating how accurate are expressive pitch contours according to 
        the unexpressive contours (the reference). Note to note mean frequency comparison. 

        Args:
            threshold (float): threshold above which the contours is considered off (in the midi norm, 0.5 is a quarter tone)
        """
        super().__init(self)
        self.threshold = threshold

    @jit(nopython=True)
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Compute the pitch loss associate to a sample from dataset

        Args:
            x (list[torch.Tensor]): sample from dataset (contours)

        Returns:
            torch.Tensor: pitch loss
        """
        # create the corresponding mask
        mask = self.__create_mask(x)

        u_f0, _ = x[0].split(1, -1)
        e_f0, _ = x[1].split(1, -1)

        # apply mask to the pitch contours
        mk_u_f0 = mask * u_f0
        mk_e_f0 = mask * e_f0

        # compute the mean frequency for both contours
        mean_u_f0 = torch.mean(mk_u_f0, dim=0) / torch.mean(mask, dim=0)
        mean_e_f0 = torch.mean(mk_e_f0, dim=0) / torch.mean(mask, dim=0)

        # compute the difference between mean frequency and loss
        diff = torch.abs(mean_u_f0 - mean_e_f0)
        loss = torch.mean((diff > self.threshold).float())

        return loss

    @jit(nopython=True)
    def __create_mask(x: List[torch.Tensor]) -> torch.Tensor:
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


class PitchLossProp(nn.Module):
    def __init__(self, threshold: .5):
        """Pitch loss estimating how accurate are expressive pitch contours according to 
        the unexpressive contours (the reference). Proportion of off points in each note pitch contour.

        Args:
            threshold (float): threshold above which the contours is considered off (in the midi norm, 0.5 is a quarter ton)
        """
        super().__init(self)
        self.threshold = threshold

    @jit(nopython=True)
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Compute the pitch loss associate to a sample from dataset. Loss computed as the proportion 
        of off pitch points for each note.

        Args:
            x (list[torch.Tensor]): sample from dataset (contours)

        Returns:
            torch.Tensor: pitch loss
        """
        # create the corresponding mask
        mask = self.__create_mask(x)

        u_f0, _ = x[0].split(1, -1)
        e_f0, _ = x[1].split(1, -1)

        # apply mask to the pitch contours
        mk_u_f0 = mask * u_f0
        mk_e_f0 = mask * e_f0

        # compute the midi pitch differences
        diff = torch.abs(mk_e_f0 - mk_u_f0)
        # proportion of contours points where pitch is wrong (diff above threshold)
        loss = torch.sum((diff > self.threshold).float()) / torch.sum(mask)

        return loss

    @jit(nopython=True)
    def __create_mask(x: List[torch.Tensor]) -> torch.Tensor:
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