import torch
import torch.nn as nn
from typing import List

from perf_gan.models.blocks.conv_blocks import ConvBlock
from perf_gan.models.blocks.linear_blocks import LinBlock


class DiscriminatorBlock(nn.Module):
    def __init__(self):
        super(DiscriminatorBlock, self).__init__()
        pass

    def forward(self, x):
        return x


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        pass