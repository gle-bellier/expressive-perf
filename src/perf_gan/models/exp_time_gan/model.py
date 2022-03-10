import torch
import torch.nn as nn

from perf_gan.models.exp_time_gan.generator import Generator
from perf_gan.models.exp_time_gan.discriminator import Discriminator

from perf_gan.models.exp_time_gan.encoder import Encoder
from perf_gan.models.exp_time_gan.decoder import Decoder


class ExpTimeGAN(nn.Module):
    def __init__(self):
        pass