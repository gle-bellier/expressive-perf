import librosa as li
import numpy as np
import torch

from typing import Union
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from perf_gan.data.descriptors import extract_lo, extract_f0


class Extractor:
    """Contours extraction tools 
    """
    def __init__(self, sr=16000, block_size=160) -> None:
        """Initialize extractor tool.

        Args:
            sr (int, optional): sampling rate. Defaults to 16000.
            block_size (int, optional): window size for f0 and loudness computing. Defaults to 160.
        """
        self.filename = filename
        self.sr = sr
        self.block_size = block_size
        self.ddsp = torch.jit.load("ddsp_flute.ts").eval()

    def extract_f0_lo(
            self, audio: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:

        lo = extract_lo(audio, self.sr, self.block_size)
        f0 = extract_f0(audio, self.sr, self.block_size)

        return f0, lo

    def reconstruct(self, f0: torch.Tensor, lo: torch.Tensor) -> np.ndarray:
        """Reconstruct sound with DDSP model according to input f0 and loudness contours

        Args:
            f0 (torch.Tensor): fundamental frequency contour
            lo (torch.Tensor): loudness contour

        Returns:
            np.ndarray: reconstructed waveform
        """
        f0 = f0.reshape(1, -1, 1)
        lo = lo.reshape(1, -1, 1)

        # synth signal with ddsp
        signal = self.ddsp(torch.tensor(f0).float(), torch.tensor(lo).float())
        return signal.reshape(-1).detach()

    def multi_scale_loss(self,
                         x: torch.Tensor,
                         resynth: np.ndarray,
                         overlap=0.75,
                         alpha=.5) -> torch.Tensor:

        loss = 0
        x = x.squeeze()
        resynth = resynth.squeeze()

        # select some window sizes for analysis
        fft_sizes = [1024, 512, 256, 128, 64]

        # compute score for each fft window size
        for fft_size in fft_sizes:
            hop = int(fft_size * (1 - overlap))
            window = torch.hann_window(fft_size).to(x)
            x_stft = torch.abs(
                torch.stft(input=x,
                           n_fft=fft_size,
                           hop_length=hop,
                           window=window,
                           normalized=True,
                           return_complex=True))
            y_stft = torch.abs(
                torch.stft(input=resynth,
                           n_fft=fft_size,
                           hop_length=hop,
                           window=window,
                           normalized=True,
                           return_complex=True))

            # Compute loss for each fft size
            loss += (y_stft - x_stft).abs().mean()
            loss += alpha * (torch.log(x_stft + 1e-7) -
                             torch.log(y_stft + 1e-7)).abs().mean()
        return loss

    def scan(self,
             x: torch.Tensor,
             resynth: np.ndarray,
             window_size=2048) -> list:
        """Compute the multi_scale_loss for each chunks of the current audio

        Args:
            x (torch.Tensor): original waveform
            resynth (np.ndarray): reconstructed waveform
            window_size (int, optional): size of each audio chunks to analyze. Defaults to 2048.

        Returns:
            list: list of the multi-scale loss each audio chunks in the sample
        """

        # we need to crop the initial sample
        # since it can be shortened during extraction
        # and reconstruction time

        x = x[:len(resynth)]
        n = len(x) // window_size
        rslt = torch.zeros(n)

        for i in range(n):
            start, end = window_size * i, window_size * (i + 1)
            rslt[i] = self.multi_scale_loss(x[start:end], resynth[start:end])

        return rslt

    def select(self,
               x: np.ndarray,
               samples_size=2048,
               ratio=0.80) -> Union[torch.Tensor, torch.Tensor]:
        """ Select the f0 and loudness contours of the best chunks of the 
        input audio according to a multi-scale spectral loss computed on the
        reconstructed audio with DDSP.

        Args:
            x (np.ndarray): input audio, raw waveform.
            samples_size (int, optional): size of each audio chunks under study. Defaults to 2048.
            ratio (float, optional): ratio of audio chunks to keep, e.g 0.7 means we 
            keep around 70% of the original audio. Defaults to 0.80.

        Returns:
            Union[torch.Tensor, torch.Tensor]: f0 and loudness contours
        """

        f0, lo = self.extract_f0_lo(x)
        resynth = self.reconstruct(f0, lo)

        losses = self.scan(torch.tensor(x), resynth, samples_size)
        # compute indices of samples to keep
        idx = torch.sort(losses)[1][:int(len(losses) * ratio)]
        # compute corresponding f0, lo chunks indices
        contours_idx = torch.floor(idx * (samples_size / self.block_size))

        select_f0, select_lo = np.empty(0), np.empty(0)
        for i in contours_idx:

            select_f0 = np.concatenate(
                (select_f0,
                 f0[int(i):int(i + (samples_size / self.block_size))]))
            select_lo = np.concatenate(
                (select_lo,
                 lo[int(i):int(i * (samples_size / self.block_size))]))

        print(f"Size ratio : {len(select_f0)/len(f0)}")
        return torch.tensor(f0), torch.tensor(lo)


path = "data/audio/"
filename = "sample1.wav"
sr = 16000

audio, fs = li.load(path + filename, sr=sr)
# audio = audio[:16000]
ext = Extractor(sr=sr)
f0, lo = ext.select(audio)
