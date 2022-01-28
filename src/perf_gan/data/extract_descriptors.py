import librosa as li
import torch
from perf_gan.data.descriptors import extract_lo, extract_f0


class Extractor:
    def __init__(self, filename, sr=16000, block_size=160):
        self.filename = filename
        self.sr = sr
        self.block_size = block_size
        self.ddsp = torch.jit.load("ddsp_flute.ts").eval()

    def extract_f0_lo(self):
        audio, fs = li.load(self.filename, sr=self.sr)
        print("Audio shape : ", audio.shape)
        lo = extract_lo(audio, self.sr, self.block_size)
        f0 = extract_f0(audio, self.sr, self.block_size)

        return f0, lo

    def reconstruct(self, f0, lo):
        f0 = f0.reshape(1, -1, 1)
        lo = lo.reshape(1, -1, 1)
        signal = self.ddsp(f0, lo)
        return signal.reshape(-1).cpu().numpy()


path = "data/audio/"
filename = "sample1.wav"

ext = Extractor(filename)
f0, lo = ext.extract_f0_lo()
print(f"f0 {f0.shape}, lo {lo.shape}")
print("reconstruct shape : ", ext.reconstruct(f0, lo).shape)