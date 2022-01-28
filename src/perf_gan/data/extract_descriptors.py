import librosa as li
import torch
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from perf_gan.data.descriptors import extract_lo, extract_f0


class Extractor:
    def __init__(self, sr=16000, block_size=160):
        self.sr = sr
        self.block_size = block_size
        self.ddsp = torch.jit.load("ddsp_flute.ts").eval()

    def extract_f0_lo(self, audio):
        lo = extract_lo(audio, self.sr, self.block_size)
        f0 = extract_f0(audio, self.sr, self.block_size)

        return f0, lo

    def reconstruct(self, f0, lo):
        f0 = f0.reshape(1, -1, 1)
        lo = lo.reshape(1, -1, 1)
        signal = self.ddsp(torch.tensor(f0).float(), torch.tensor(lo).float())
        return signal.reshape(-1).detach()

    def multi_scale_loss(self, x, resynth, overlap=0.75, alpha=1):
        loss = 0
        x = x.squeeze()
        resynth = resynth.squeeze()
        fft_sizes = [1024, 512, 256, 128, 64]
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

    def scan(self, x, resynth, window_size=2048):
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


path = "data/audio/"
filename = "sample1.wav"
sr = 16000

audio, fs = li.load(path + filename, sr=sr)
#audio = audio[:16000]
ext = Extractor(sr=sr)
f0, lo = ext.extract_f0_lo(audio)
resynth = ext.reconstruct(f0, lo)

length = 8096
loss = ext.scan(torch.tensor(audio), resynth)
print("Loss : ", loss)